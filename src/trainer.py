# src/trainer.py
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
import os
import torch.distributed as dist
from src.utils import get_scheduler

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, rank, world_size):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}")
        
        self.output_dir = self.config['saving']['output_dir']
        self.best_model_path = os.path.join(self.output_dir, self.config['saving']['best_model_name'])

        self.classification_criterion = nn.CrossEntropyLoss().to(self.device)

        self.N_supervision = self.config['recursion']['N_supervision_steps'] if self.config['recursion']['recursive_mode'] else 1
        self.N_latent = self.config['recursion']['N_latent_steps']
        self.N_deep = self.config['recursion']['N_deep_steps']

        # Early stopping setup
        self.early_stop_config = self.config['training']['early_stopping']
        self.early_stopping_enabled = self.early_stop_config['enabled']
        
        if self.early_stopping_enabled:
            self.es_patience = self.early_stop_config['patience']
            self.es_metric = self.early_stop_config['metric']
            self.es_mode = self.early_stop_config['mode']
            self.es_counter = 0
            self.best_metric_value = float('inf') if self.es_mode == 'min' else float('-inf')
            if self.rank == 0:
                print(f"Early stopping enabled: monitor='{self.es_metric}', mode='{self.es_mode}', patience={self.es_patience}")
        else:
            self.es_patience = None
            self.es_metric = 'val_loss'
            self.es_mode = 'min'
            self.es_counter = 0
            self.best_metric_value = float('inf')
            if self.rank == 0:
                print("Early stopping disabled")

        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        lr = self.config['training']['learning_rate']
        if self.config['training']['scale_lr_by_world_size']:
            lr *= self.world_size
            if self.rank == 0:
                print(f"Scaled learning rate to: {lr}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )
        
        num_training_steps_per_epoch = len(self.train_loader)
        num_training_steps = num_training_steps_per_epoch * self.config['training']['epochs']

        self.scheduler = get_scheduler(
            name=self.config['training']['lr_scheduler_type'],
            optimizer=self.optimizer,
            num_training_steps=num_training_steps
        )

    def train_epoch(self):
        """
        Training epoch implementing the deeply-supervised recursive loop.
        """
        self.model.train()
        total_loss_sum = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0, device=self.device)
        
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            B = images.shape[0]

            # --- UPDATED INITIALIZATION ---
            # Handle STATE_DIM being a tuple (ViT) or int (CNN)
            state_dim = self.model.module.STATE_DIM
            if isinstance(state_dim, int):
                init_shape = (B, state_dim)
            else:
                # state_dim is a tuple, e.g., (Seq_Len, Dim)
                init_shape = (B, *state_dim)

            if self.config['recursion']['init_strategy'] == "zeros":
                output_embed = torch.zeros(*init_shape, device=self.device)
                latent_embed = torch.zeros(*init_shape, device=self.device)
            elif self.config['recursion']['init_strategy'] == "random":
                output_embed = torch.randn(*init_shape, device=self.device)
                latent_embed = torch.randn(*init_shape, device=self.device)
            # -------------------------------
            
            batch_total_loss = 0.0

            for step in range(self.N_supervision):
                self.optimizer.zero_grad()
                
                output_embed_detached = output_embed.detach()
                latent_embed_detached = latent_embed.detach()

                output_embed, latent_embed, logits = self.model.module.deep_recursion(
                    images, 
                    output_embed_detached, 
                    latent_embed_detached,
                    n=self.N_latent,
                    T=self.N_deep
                )

                class_loss = self.classification_criterion(logits, labels)
                
                step_loss = class_loss
                step_loss.backward()
                self.optimizer.step()
                
                batch_total_loss += step_loss.detach()
            
            self.scheduler.step()

            total_loss_sum += (batch_total_loss / self.N_supervision)
            num_batches += 1
        
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_sum.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0, device=self.device)

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model.module.forward(
                images, 
                Nsup=self.N_supervision, 
                n_latent=self.N_latent
            )
            
            loss = self.classification_criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_loss += loss.detach()
            total_correct += (predicted == labels).sum()
            total_samples += labels.size(0)
            num_batches += 1
        
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        accuracy = total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0
        
        return avg_loss, accuracy

    def train(self):
        epochs = self.config['training']['epochs']
        
        progress_bar = tqdm(
            total=epochs,
            desc="Overall Training Progress",
            disable=(self.rank != 0)
        )
        
        for epoch in range(epochs):
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.valid_epoch()

            if self.rank == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
                
                is_best = self.check_best_metric(val_loss, val_accuracy)
                if is_best:
                    print(f"New best metric achieved. Saving model to {self.best_model_path}")
                    self.save_best_model()
                    if self.early_stopping_enabled:
                        self.es_counter = 0
                else:
                    if self.early_stopping_enabled:
                        self.es_counter += 1
                        print(f"No improvement. Early stopping patience: {self.es_counter}/{self.es_patience}")
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_accuracy:.4f}'
                })
                
            dist.barrier()

            if self.early_stopping_enabled:
                stop_signal = torch.tensor([0], dtype=torch.long, device=self.device)
                if self.rank == 0:
                    if self.es_counter >= self.es_patience:
                        stop_signal[0] = 1
                
                dist.broadcast(stop_signal, src=0)
                
                if stop_signal.item() == 1:
                    if self.rank == 0:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                        progress_bar.close()
                    break
        
        if self.rank == 0:
            progress_bar.close()

    def check_best_metric(self, val_loss, val_accuracy):
        if self.es_metric == 'val_loss':
            current_value = val_loss
        elif self.es_metric == 'val_accuracy':
            current_value = val_accuracy
        else:
            raise ValueError(f"Unknown metric for best model tracking: {self.es_metric}")

        if self.es_mode == 'min':
            if current_value < self.best_metric_value:
                self.best_metric_value = current_value
                return True
        elif self.es_mode == 'max':
            if current_value > self.best_metric_value:
                self.best_metric_value = current_value
                return True
        return False

    def save_best_model(self):
        if self.config['saving']['save_only_on_rank_0'] and self.rank != 0:
            return

        torch.save(self.model.module.state_dict(), self.best_model_path)
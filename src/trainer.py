# src/trainer.py
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os
import torch.distributed as dist
import json

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, rank, world_size):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}")
        
        self.best_metric_value = float('inf') if self.config['training']['early_stopping']['mode'] == 'min' else float('-inf')
        self.output_dir = self.config['saving'].get('output_dir', './experiments')
        self.best_model_path = os.path.join(self.output_dir, self.config['saving']['best_model_name'])

        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss().to(self.device)
        # self.quality_criterion = nn.BCEWithLogitsLoss().to(self.device) # For q_hat <-- REMOVED

        # Recursion HParams
        self.N_supervision = self.config['recursion']['N_supervision_steps']
        self.N_latent = self.config['recursion']['N_latent_steps']
        self.N_deep = self.config['recursion']['N_deep_steps']
        # self.quality_weight = self.config['training']['quality_loss_weight'] <-- REMOVED
        self.Nsup_inference = self.config['testing']['Nsup']

        # Early stopping setup
        self.early_stop_config = self.config['training'].get('early_stopping', {'enabled': False})
        if self.early_stop_config.get('enabled', False):
            # ... (early stopping setup identical to previous file) ...
            self.es_patience = self.early_stop_config.get('patience', 3)
            self.es_metric = self.early_stop_config.get('metric', 'val_loss')
            self.es_mode = self.early_stop_config.get('mode', 'min')
            self.es_counter = 0
            if self.rank == 0:
                print(f"Early stopping enabled: monitor='{self.es_metric}', mode='{self.es_mode}', patience={self.es_patience}")

        # Move model to the correct device and wrap with DDP
        self.model.to(self.device)
        # find_unused_parameters=True is crucial for DDP when parts of the model
        # (like the 'forward' method) are not used in the training pass.
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        # Optimizer and Scheduler
        lr = self.config['training']['learning_rate']
        if self.config['training']['scale_lr_by_world_size']:
            lr *= self.world_size
            if self.rank == 0:
                print(f"Scaled learning rate to: {lr} (base: {self.config['training']['learning_rate']} * world_size: {self.world_size})")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        # Num steps is now per-epoch, as scheduler steps once per batch
        num_training_steps_per_epoch = len(self.train_loader)
        num_training_steps = num_training_steps_per_epoch * self.config['training']['epochs']

        self.scheduler = get_scheduler(
            name=self.config['training']['lr_scheduler_type'],
            optimizer=self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
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

            # Initialize recursive states (y, z)
            # We use randn, (B, 128) is the feature dim from the model
            output_embed = torch.randn(B, 128, device=self.device)
            latent_embed = torch.randn(B, 128, device=self.device)
            
            batch_total_loss = 0.0

            # --- Start Deep Supervision Loop (from pseudo-code) ---
            for step in range(self.N_supervision):
                self.optimizer.zero_grad()
                
                # Detach states to prevent gradients from flowing back >1 step
                output_embed_detached = output_embed.detach()
                latent_embed_detached = latent_embed.detach()

                # Call the model's recursive function
                # We must call .module. to bypass the DDP wrapper
                output_embed, latent_embed, logits = self.model.module.deep_recursion( # <-- CHANGED
                    images, 
                    output_embed_detached, 
                    latent_embed_detached,
                    n=self.N_latent,
                    T=self.N_deep
                )

                # 1. Classification Loss (y_hat vs y_true)
                class_loss = self.classification_criterion(logits, labels)
                
                # 2. Quality Loss (q_hat vs (y_hat == y_true)) <-- SECTION REMOVED
                
                # Total loss for this step
                step_loss = class_loss # + self.quality_weight * quality_loss <-- CHANGED
                
                # Backward pass *per step*
                step_loss.backward()
                self.optimizer.step()
                
                batch_total_loss += step_loss.detach()
            # --- End Deep Supervision Loop ---
            
            # Step the scheduler *once per batch* (after all supervision steps)
            self.scheduler.step()

            total_loss_sum += (batch_total_loss / self.N_supervision) # Avg loss for the batch
            num_batches += 1
        
        # Reduce metrics
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_sum.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def valid_epoch(self):
        """
        Validation epoch. Uses the model's inference-only .forward() method.
        """
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0, device=self.device)

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Use the inference-only forward pass
            # We must call .module. to bypass the DDP wrapper
            outputs = self.model.module.forward(
                images, 
                Nsup=self.Nsup_inference, 
                n_latent=self.N_latent
            )
            
            loss = self.classification_criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_loss += loss.detach()
            total_correct += (predicted == labels).sum()
            total_samples += labels.size(0)
            num_batches += 1
        
        # Reduce metrics
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        accuracy = total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0
        
        return avg_loss, accuracy

    def train(self):
        # ... (This function is identical to the previous version) ...
        epochs = self.config['training']['epochs']
        
        for epoch in range(epochs):
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.valid_epoch()

            # Rank 0 handles logging, saving, and early stopping decisions
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
                
                # Checkpointing
                is_best = self.check_best_metric(val_loss, val_accuracy)
                if is_best:
                    print(f"New best metric achieved. Saving model to {self.best_model_path}")
                    self.save_best_model()
                    self.es_counter = 0 # Reset early stopping counter
                else:
                    if self.early_stop_config.get('enabled', False):
                        self.es_counter += 1
                        print(f"No improvement. Early stopping patience: {self.es_counter}/{self.es_patience}")
                
            # Synchronize all processes before checking early stopping
            dist.barrier()

            # Broadcast early stopping decision from rank 0
            if self.early_stop_config.get('enabled', False):
                stop_signal = torch.tensor([0], dtype=torch.long, device=self.device)
                if self.rank == 0:
                    if self.es_counter >= self.es_patience:
                        stop_signal[0] = 1 # Signal to stop
                
                dist.broadcast(stop_signal, src=0)
                
                if stop_signal.item() == 1:
                    if self.rank == 0:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break

    def check_best_metric(self, val_loss, val_accuracy):
        # ... (This function is identical to the previous version) ...
        if self.es_metric == 'val_loss':
            current_value = val_loss
        elif self.es_metric == 'val_accuracy':
            current_value = val_accuracy
        else:
            raise ValueError(f"Unknown early stopping metric: {self.es_metric}")

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
        # ... (This function is identical to the previous version) ...
        if self.config['saving']['save_only_on_rank_0'] and self.rank != 0:
            return

        torch.save(self.model.module.state_dict(), self.best_model_path)
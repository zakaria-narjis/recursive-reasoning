import yaml
import torch
import os
from datetime import datetime
import shutil
import torch.distributed as dist

from model import SimpleCNN
from dataset import get_dataloaders
from trainer import Trainer
from test import test_model

def main(config):
    """
    Main function to set up DDP, load components, train, and test.
    """
    # --- 1. Setup DDP ---
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("DDP environment variables not set. Running in single-process mode.")
        rank = 0
        world_size = 1
        local_rank = 0
        # Set dummy env vars for compatibility
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if is_ddp:
        dist.init_process_group(config['ddp']['backend'])
        if rank == 0:
             print(f"DDP initialized with backend '{config['ddp']['backend']}' and world size {world_size}")
    else:
        if rank == 0:
            print("Running in single-process mode.")

    # --- 2. Create Output Directory (Rank 0 only) ---
    output_dir = None
    if rank == 0:
        model_name = "SimpleCNN_MNIST"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"{model_name}_{timestamp}"
        output_dir = os.path.join(config['saving']['base_output_dir'], experiment_name)
        
        config['saving']['output_dir'] = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the config file for reproducibility
        try:
            with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
                yaml.dump(config, f)
            print(f"Experiment outputs will be saved to: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not save config file. Error: {e}")
    
    # Broadcast output_dir to all ranks
    output_dir_list = [output_dir]
    if is_ddp:
        dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]
    
    # Update config on all ranks
    config['saving']['output_dir'] = output_dir

    # --- 3. Load Model ---
    if rank == 0:
        print("Loading model...")
    model_config = config['model']
    model = SimpleCNN(
        num_classes=model_config['num_classes'],
        channels_in=model_config['channels_in'],
        channels_l1=model_config['channels_l1'],
        channels_l2=model_config['channels_l2']
    )
    # Model is moved to device inside Trainer
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created. Trainable parameters: {total_params:,}")


    # --- 4. Load Data ---
    if rank == 0:
        print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config, rank, world_size)
    if rank == 0:
        print(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # --- 5. Start Training ---
    if rank == 0:
        print("Initializing trainer...")
    trainer = Trainer(model, config, train_loader, val_loader, rank, world_size)
    
    if rank == 0:
        print("Starting training...")
    trainer.train()
    if rank == 0:
        print("Training finished.")

    # --- 6. Synchronize all processes before testing ---
    del model # Release model from trainer
    del trainer
    torch.cuda.empty_cache()
    if is_ddp:
        dist.barrier()
    
    # --- 7. Run Testing ---
    # We can run testing on all ranks, the test_model function handles aggregation
    if rank == 0:
        print("Starting final evaluation on test set...")
    test_model(config, output_dir, device, rank, world_size)
        
    # --- 8. Final barrier and cleanup ---
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("Experiment complete.")


if __name__ == "__main__":
    # Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
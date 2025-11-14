# train.py
import yaml 
import torch
import os
import random
import numpy as np
import argparse
from datetime import datetime
import torch.distributed as dist

from model import SimpleCNN, ResNet
from dataset import get_dataloaders, get_dataset_stats
from trainer import Trainer
from test import test_model

def set_seed(seed):
    """Set seeds for reproducibility across all ranks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def configure_model_from_dataset(config):
    """
    Automatically configure model parameters based on the selected dataset.
    Updates config['model'] with the appropriate in_channels and num_classes.
    """
    dataset_name = config['data']['dataset_name']
    dataset_stats = get_dataset_stats(dataset_name)
    
    # Update model config with dataset-specific parameters
    config['model']['in_channels'] = dataset_stats['in_channels']
    config['model']['num_classes'] = dataset_stats['num_classes']
    config['model']['input_size'] = dataset_stats['input_size']
    return config

def main(config):
    """
    Main function to set up DDP, load components, train, and test.
    This version always initializes the process group and fixes the seed.
    """
    # --- 0. Fix Seed ---
    if 'seed' in config:
        set_seed(config['seed'])
        if int(os.environ["RANK"]) == 0:
            print(f"Fixed seed set to: {config['seed']}")
    else:
        if int(os.environ["RANK"]) == 0:
            print("Warning: 'seed' key not found in config. Running without fixed seed.")
    
    # --- 0.5. Configure model based on dataset ---
    config = configure_model_from_dataset(config)
    if int(os.environ["RANK"]) == 0:
        print(f"Dataset: {config['data']['dataset_name']}")
        print(f"Model configured: in_channels={config['model']['in_channels']}, num_classes={config['model']['num_classes']}, input_size={config['model']['input_size']}")
        print(f"Data augmentation: {'enabled' if config['data'].get('augmentation', True) else 'disabled'}")
            
    # --- 1. Setup DDP ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Always initialize the process group, even if world_size=1
    dist.init_process_group(config['ddp']['backend'])
    
    if rank == 0:
        print(f"DDP initialized with backend '{config['ddp']['backend']}' and world size {world_size}")

    # --- 2. Create Output Directory (Rank 0 only) ---
    output_dir = None
    if rank == 0:
        model_name = f"{config['model']['name']}_{config['data']['dataset_name']}"
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
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]
    
    # Update config on all ranks
    config['saving']['output_dir'] = output_dir

    # --- 3. Load Model ---
    if rank == 0:
        print("Loading model...")    
        model_config = config['model']
        if model_config["name"] == "SimpleCNN":
            model = SimpleCNN(
                    in_channels=model_config['in_channels'],
                    num_classes=model_config['num_classes'],
                    input_size=model_config['input_size'],
                    recursive_mode=config['recursion']['recursive_mode'],
            )
        elif model_config["name"] == "ResNet":
            model = ResNet(
                num_classes=model_config['num_classes'],
                recursive_mode=config['recursion']['recursive_mode'],
                pretrained=model_config['pretrained'],
                use_precomputed_features=config['data']['use_precomputed_features']
            )
        else:
            raise ValueError(f"Unsupported model name: {model_config['name']}")
    if rank == 0:
        print(f"Using precomputed features: {config['data']['use_precomputed_features']}")
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
    del model
    del trainer
    torch.cuda.empty_cache()
    dist.barrier()
    
    # --- 7. Run Testing ---
    if rank == 0:
        print("Starting final evaluation on test set...")
    test_model(test_loader, config, output_dir, device, rank, world_size)
        
    # --- 8. Final barrier and cleanup ---
    dist.barrier()
    dist.destroy_process_group()
    
    if rank == 0:
        print("Experiment complete.")

# Run with: CUDA_VISIBLE_DEVICES="0" torchrun --standalone --nproc_per_node=1 src/train.py --config config/config.yaml
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with specified configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file (default: config/config.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
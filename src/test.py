# src/test.py
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.distributed as dist
import os
import json

# We need to import the model definition to instantiate it
from model import SimpleCNN

@torch.no_grad()
def test_model(test_loader, config, output_dir, device, rank, world_size):
    if rank == 0:
        print("\n" + "="*80)
        print("Starting Final Evaluation on Test Set")
        print("="*80)

    best_model_path = os.path.join(output_dir, config['saving']['best_model_name'])
    if not os.path.exists(best_model_path):
        if rank == 0:
            print(f"Error: Best model not found at {best_model_path}. Cannot run test.")
        return

    # 1. Instantiate the model on the correct device
    model_config = config['model']
    model = SimpleCNN(
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes'],
        input_size=model_config['input_size'],
    )
    
    # 2. Load the saved state_dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if 'cuda' in device.type else device
    model.load_state_dict(torch.load(best_model_path, map_location=map_location))
    model.to(device)
    
    # 3. Wrap model in DDP for distributed evaluation
    # find_unused_parameters=True is good practice if different paths
    # are taken in forward() vs. other methods.
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)

    # Get inference hyperparameters
    Nsup_test = config.get('testing', {}).get('Nsup', 10)
    n_latent = config['recursion']['N_latent_steps']
    if rank == 0:
        print(f"Running inference with Nsup={Nsup_test}, n_latent={n_latent}")

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0, device=device)

    if rank == 0:
        progress_bar = tqdm(test_loader, desc=f"Testing (Rank {rank})")
    else:
        progress_bar = test_loader

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # --- MODIFIED MODEL CALL ---
        # Call the inference-only forward method
        # No .module. needed here as we are in no_grad() and .eval()
        # but it's safer to call .module. to be consistent
        outputs = model.module.forward(images, Nsup=Nsup_test, n_latent=n_latent)
        # --- END MODIFICATION ---
        
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total_loss += loss.detach()
        total_correct += (predicted == labels).sum()
        total_samples += labels.size(0)
        num_batches += 1

    # ... (Rest of the file is identical to the previous version) ...
    # Synchronize all processes
    dist.barrier()

    # Reduce metrics
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

    # Rank 0 calculates and prints final results
    if rank == 0:
        avg_loss = total_loss.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        accuracy = total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0

        print("\n" + "="*80)
        print("Test Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy:     {accuracy:.4f} ({int(total_correct.item())}/{int(total_samples.item())})")
        print("="*80)

        # Save results to a JSON file
        results = {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "correct_samples": int(total_correct.item()),
            "total_samples": int(total_samples.item())
        }
        test_results_path = os.path.join(output_dir, "test_results.json")
        try:
            with open(test_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Test results saved to {test_results_path}")
        except Exception as e:
            print(f"Failed to save test results: {e}")

    dist.barrier()
# src/test.py
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.distributed as dist
import os
import json

from model import SimpleCNN, ResNet


@torch.no_grad()
def test_model(test_loader, config, output_dir, device, rank, world_size):
    """
    Test the model with multiple N_supervision_steps values.
    Saves separate results for each N_supervision value.
    """
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
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if 'cuda' in device.type else device
    model.load_state_dict(torch.load(best_model_path, map_location=map_location))
    model.to(device)
    
    # 3. Wrap model in DDP for distributed evaluation
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)

    # Get test configurations
    n_latent = config['recursion']['N_latent_steps']
    init_strategy = config['recursion']['init_strategy']
    recursive_mode = config['recursion']['recursive_mode']
    
    # Determine which N_supervision values to test
    if recursive_mode:
        # Test with multiple N_supervision values for recursive mode
        test_nsup_values = config["testing"]["N_supervision_steps"] if config['recursion']['recursive_mode'] else 1
        # Ensure it's a list
        if not isinstance(test_nsup_values, list):
            test_nsup_values = [test_nsup_values]
    else:
        # For non-recursive mode, only test with N_supervision=1
        test_nsup_values = [1]
    
    if rank == 0:
        print(f"Testing with N_supervision_steps values: {test_nsup_values}")
        print(f"n_latent={n_latent}, init_strategy={init_strategy}")
    
    # Store all results
    all_results = {}
    
    # Test with each N_supervision value
    for Nsup_test in test_nsup_values:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing with N_supervision_steps = {Nsup_test}")
            print(f"{'='*60}")
        
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(0, device=device)

        if rank == 0:
            progress_bar = tqdm(test_loader, desc=f"Testing Nsup={Nsup_test} (Rank {rank})")
        else:
            progress_bar = test_loader

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Run inference with current N_supervision value
            outputs = model.module.forward(
                images, 
                Nsup=Nsup_test, 
                n_latent=n_latent, 
                init_strategy=init_strategy
            )
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total_loss += loss.detach()
            total_correct += (predicted == labels).sum()
            total_samples += labels.size(0)
            num_batches += 1

        # Synchronize all processes
        dist.barrier()

        # Reduce metrics across all ranks
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

        # Rank 0 calculates and stores results
        if rank == 0:
            avg_loss = total_loss.item() / num_batches.item() if num_batches.item() > 0 else 0.0
            accuracy = total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0

            print(f"\nResults for N_supervision_steps = {Nsup_test}:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy:     {accuracy:.4f} ({int(total_correct.item())}/{int(total_samples.item())})")

            # Store results for this N_supervision value
            all_results[f"Nsup_{Nsup_test}"] = {
                "N_supervision_steps": Nsup_test,
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
                "correct_samples": int(total_correct.item()),
                "total_samples": int(total_samples.item())
            }

        dist.barrier()
    
    # Save all results (rank 0 only)
    if rank == 0:
        print("\n" + "="*80)
        print("Test Results Summary:")
        print("="*80)
        
        # Find best performing N_supervision
        best_nsup = max(all_results.keys(), key=lambda k: all_results[k]['test_accuracy'])
        best_acc = all_results[best_nsup]['test_accuracy']
        
        print(f"Best performance: Nsup={all_results[best_nsup]['N_supervision_steps']} "
              f"with accuracy {best_acc:.4f}")
        
        # Save detailed results
        test_results_path = os.path.join(output_dir, "test_results.json")
        try:
            # Add metadata
            results_with_metadata = {
                "metadata": {
                    "recursive_mode": recursive_mode,
                    "n_latent_steps": n_latent,
                    "init_strategy": init_strategy,
                    "best_N_supervision": all_results[best_nsup]['N_supervision_steps'],
                    "best_accuracy": best_acc
                },
                "results_by_Nsup": all_results
            }
            
            with open(test_results_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
            print(f"\nDetailed test results saved to {test_results_path}")
        except Exception as e:
            print(f"Failed to save test results: {e}")
        
        # Also save a simplified version for backward compatibility
        # Use the best N_supervision result as the default
        simple_results = {
            "test_loss": all_results[best_nsup]['test_loss'],
            "test_accuracy": all_results[best_nsup]['test_accuracy'],
            "N_supervision_steps_used": all_results[best_nsup]['N_supervision_steps'],
            "all_Nsup_results": {k: v['test_accuracy'] for k, v in all_results.items()}
        }
        
        simple_results_path = os.path.join(output_dir, "test_results_simple.json")
        with open(simple_results_path, 'w') as f:
            json.dump(simple_results, f, indent=2)
        
        print("="*80)

    dist.barrier()
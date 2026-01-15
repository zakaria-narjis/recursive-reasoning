# src/utils.py
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, SequentialLR, LinearLR

def get_scheduler(name, optimizer, num_training_steps, warmup_steps=0, **kwargs):
    name = name.lower()
    
    if name == "none":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    
    # Define the main scheduler
    if name == "cosine":
        # Note: T_max should be the remaining steps after warmup
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - warmup_steps, 
            eta_min=kwargs.get('eta_min', 0)
        )
    elif name == "step":
        step_size = kwargs.get('step_size', num_training_steps // 10)
        gamma = kwargs.get('gamma', 0.1)
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    # If no warmup is needed, return the main scheduler
    if warmup_steps <= 0:
        return main_scheduler

    # Create warmup scheduler (linearly increases LR from 0 to Initial LR)
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-8, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )

    # Chain them together
    # 1. Run warmup_scheduler for `warmup_steps`
    # 2. Then run main_scheduler
    return SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )
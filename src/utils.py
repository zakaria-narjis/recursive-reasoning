# src/scheduler_utils.py
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR


def get_scheduler(name, optimizer, num_training_steps, **kwargs):
    """
    Create a learning rate scheduler without warmup.
    
    Args:
        name: Type of scheduler ('cosine', 'step', 'none')
        optimizer: PyTorch optimizer
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments for specific schedulers
            - step_size: for StepLR (default: num_training_steps // 10)
            - gamma: for StepLR (default: 0.1)
            - eta_min: for CosineAnnealingLR (default: 0)
    
    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    name = name.lower()
    
    if name == "none":
        # Constant learning rate (no scheduling)
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    
    elif name == "cosine":
        # Cosine annealing
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps,
            eta_min=eta_min
        )
    
    elif name == "step":
        # Step LR
        step_size = kwargs.get('step_size', num_training_steps // 10)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    else:
        raise ValueError(
            f"Unknown scheduler type: {name}. "
            f"Supported types: 'cosine', 'step', 'none'"
        )

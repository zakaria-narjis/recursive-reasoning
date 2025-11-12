# src/dataset.py
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import os

def get_dataloaders(config, rank, world_size,seed=42):
    """
    Creates training, validation, and test dataloaders for MNIST.
    """
    data_path = config['data']['path']
    val_split_size = config['data']['val_split_size']

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])

    # Download and load training data
    # Only download on rank 0 to avoid race conditions
    if rank == 0:
        full_train_dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    
    # Wait for rank 0 to finish downloading
    torch.distributed.barrier()
    
    # Other ranks load the downloaded data
    if rank != 0:
        full_train_dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=False, # Data is already downloaded by rank 0
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_path,
            train=False,
            download=False, # Data is already downloaded by rank 0
            transform=transform
        )

    # Split training data into train and validation
    dataset_size = len(full_train_dataset)
    val_size = int(val_split_size * dataset_size)
    train_size = dataset_size - val_size

    # Use a fixed generator for reproducible splits
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Dataloader worker configuration
    worker_config = config['data'].get('dataloader_workers', {})
    train_workers = worker_config.get('train', 4)
    val_workers = worker_config.get('val', 2)
    test_workers = worker_config.get('test', 2)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=train_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=val_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        sampler=test_sampler,
        num_workers=test_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
# src/dataset.py
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import os

def get_dataset_stats(dataset_name):
    """
    Returns normalization statistics (mean, std) and number of classes for common datasets.
    """
    stats = {
        'MNIST': {
            'mean': (0.1307,),
            'std': (0.3081,),
            'num_classes': 10,
            'in_channels': 1,
            'input_size': (28, 28) # <--- ADDED
        },
        'FashionMNIST': {
            'mean': (0.2860,),
            'std': (0.3530,),
            'num_classes': 10,
            'in_channels': 1,
            'input_size': (28, 28) # <--- ADDED
        },
        'CIFAR10': {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (32, 32) # <--- ADDED
        },
        'CIFAR100': {
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'num_classes': 100,
            'in_channels': 3,
            'input_size': (32, 32) # <--- ADDED
        },
        'SVHN': {
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (32, 32) # <--- ADDED
        },
        'STL10': {
            'mean': (0.4467, 0.4398, 0.4066),
            'std': (0.2603, 0.2566, 0.2713),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (96, 96) # <--- ADDED
        },
        'KMNIST': {
            'mean': (0.1918,),
            'std': (0.3483,),
            'num_classes': 10,
            'in_channels': 1,
            'input_size': (28, 28) # <--- ADDED
        }
    }
    
    if dataset_name not in stats:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(stats.keys())}")
    
    return stats[dataset_name]

def get_transforms(dataset_name, augment=True):
    """
    Returns train and test transforms based on the dataset.
    
    Args:
        dataset_name: Name of the dataset
        augment: Whether to apply data augmentation for training
    """
    stats = get_dataset_stats(dataset_name)
    mean = stats['mean']
    std = stats['std']
    in_channels = stats['in_channels']
    
    # Base transforms for grayscale datasets (MNIST, FashionMNIST, KMNIST)
    if in_channels == 1:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # RGB datasets (CIFAR10, CIFAR100, SVHN, STL10)
    else:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4) if dataset_name in ['CIFAR10', 'CIFAR100'] else transforms.RandomCrop(96, padding=12) if dataset_name == 'STL10' else transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return train_transform, test_transform

def load_dataset(dataset_name, data_path, train, download, transform):
    """
    Loads the specified dataset from torchvision.
    """
    dataset_class = getattr(datasets, dataset_name)
    
    # Special handling for SVHN which uses 'split' instead of 'train'
    if dataset_name == 'SVHN':
        split = 'train' if train else 'test'
        return dataset_class(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    # Special handling for STL10
    elif dataset_name == 'STL10':
        split = 'train' if train else 'test'
        return dataset_class(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    else:
        return dataset_class(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )

# =================================================================
# === THIS CLASS IS UNCHANGED ===
# =================================================================
class PrecomputedFeatureDataset(Dataset):
    """
    A dataset class that loads precomputed features and labels from a .pt file.
    """
    def __init__(self, file_path, rank=0):
        if rank == 0:
            print(f"Loading precomputed data from {file_path}...")
        data = torch.load(file_path, map_location='cpu') # Load on CPU to save GPU memory
        self.features = data['features']
        self.labels = data['labels']
        if rank == 0:
            print(f"Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return precomputed feature and its label
        return self.features[idx], self.labels[idx]
# =================================================================


def get_dataloaders(config, rank, world_size, seed=42):
    """
    Creates training, validation, and test dataloaders for the specified dataset.
    
    Args:
        config: Configuration dictionary
        rank: Current process rank
        world_size: Total number of processes
        seed: Random seed for reproducibility
    """
    data_path = config['data']['path']
    dataset_name = config['data']['dataset_name']
    val_split_size = config['data']['val_split_size']
    use_augmentation = config['data'].get('augmentation', True)
    
    # --- Check for precomputed feature usage ---
    use_precomputed = config['data'].get('use_precomputed_features', False)
    
    # --- THIS IS THE FIX ---
    # Get the base path from config, default to 'scratch/narjis'
    base_path = config['data'].get('precomputed_path', 'scratch/narjis') 
    # --- END OF FIX ---

    if use_precomputed:
        if config['model']['name'] != 'ResNet' or not config['model']['pretrained']:
            if rank == 0:
                raise ValueError("use_precomputed_features is True, but model is not 'ResNet' with 'pretrained=True'")
        
        # --- THIS IS THE FIX ---
        # Construct the full path to the embedding directory
        # e.g., scratch/narjis/CIFAR10/precomputed embeding
        embedding_dir = os.path.join(base_path, dataset_name, "precomputed_embeding")

        if rank == 0:
            print(f"Using precomputed features from: {embedding_dir}")
        
        # Construct the full file paths
        train_file = os.path.join(embedding_dir, "train.pt")
        test_file = os.path.join(embedding_dir, "test.pt")
        # --- END OF FIX ---

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            # This error message will now show the correct path
            raise FileNotFoundError(f"Precomputed files not found at {embedding_dir}. Run src/precompute.py first.")
        
        # All ranks load directly from the files
        full_train_dataset = PrecomputedFeatureDataset(train_file, rank)
        test_dataset = PrecomputedFeatureDataset(test_file, rank)
        
        # NOTE: Augmentation is skipped when using precomputed features.
        if use_augmentation and rank == 0:
            print("Warning: Data augmentation is disabled when using precomputed features.")

    else:
        # --- Original logic for loading raw images ---
        if rank == 0:
            print("Loading raw image data...")
        train_transform, test_transform = get_transforms(dataset_name, augment=use_augmentation)
        
        # Download and load training data (only on rank 0)
        if rank == 0:
            full_train_dataset = load_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = load_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                train=False,
                download=True,
                transform=test_transform
            )
        
        # Wait for rank 0 to finish downloading
        torch.distributed.barrier()
        
        # Other ranks load the downloaded data
        if rank != 0:
            full_train_dataset = load_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                train=True,
                download=False,
                transform=train_transform
            )
            test_dataset = load_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                train=False,
                download=False,
                transform=test_transform
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
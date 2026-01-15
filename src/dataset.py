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
            'input_size': (28, 28),
            'has_split': True
        },
        'FashionMNIST': {
            'mean': (0.2860,),
            'std': (0.3530,),
            'num_classes': 10,
            'in_channels': 1,
            'input_size': (28, 28),
            'has_split': True
        },
        'CIFAR10': {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (32, 32),
            'has_split': True
        },
        'CIFAR100': {
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'num_classes': 100,
            'in_channels': 3,
            'input_size': (32, 32),
            'has_split': True
        },
        'SVHN': {
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (32, 32),
            'has_split': True
        },
        'STL10': {
            'mean': (0.4467, 0.4398, 0.4066),
            'std': (0.2603, 0.2566, 0.2713),
            'num_classes': 10,
            'in_channels': 3,
            'input_size': (96, 96),
            'has_split': True
        },
        'KMNIST': {
            'mean': (0.1918,),
            'std': (0.3483,),
            'num_classes': 10,
            'in_channels': 1,
            'input_size': (28, 28),
            'has_split': True
        },
        'ImageNet': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 1000,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': True
        },
        'TinyImageNet': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 200,
            'in_channels': 3,
            'input_size': (64, 64),
            'has_split': True
        },
        'Caltech101': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 101,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': False,  # No built-in split
            'test_split': 0.2  # 20% for test
        },
        'Caltech256': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 257,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': False,  # No built-in split
            'test_split': 0.2
        },
        'Food101': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 101,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': True
        },
        'Places365': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 365,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': True
        },
        'Flowers102': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 102,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': True  # Actually has train/val/test splits
        },
        'OxfordIIITPet': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 37,
            'in_channels': 3,
            'input_size': (224, 224),
            'has_split': True  # Has trainval and test splits
        }
    }
    
    if dataset_name not in stats:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(stats.keys())}")
    
    return stats[dataset_name]

def get_transforms(dataset_name, augment=True):
    """
    Returns train and test transforms based on the dataset.
    Ensures images are converted to RGB for 3-channel datasets.
    """
    stats = get_dataset_stats(dataset_name)
    mean = stats['mean']
    std = stats['std']
    in_channels = stats['in_channels']
    input_size = stats['input_size']
    
    # Helper to force RGB conversion
    to_rgb = transforms.Lambda(lambda x: x.convert('RGB'))
    
    # 1. Grayscale datasets
    if in_channels == 1:
        to_gray = transforms.Lambda(lambda x: x.convert('L'))
        
        if augment:
            train_transform = transforms.Compose([
                to_gray,
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                to_gray,
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            to_gray,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # 2. High-resolution RGB datasets
    elif dataset_name in ['ImageNet', 'TinyImageNet', 'Caltech101', 'Caltech256', 
                          'Food101', 'Places365', 'Flowers102', 'OxfordIIITPet']:
        if augment:
            train_transform = transforms.Compose([
                to_rgb,
                transforms.RandomResizedCrop(input_size[0], scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                to_rgb,
                transforms.Resize(256),
                transforms.CenterCrop(input_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            to_rgb,
            transforms.Resize(256),
            transforms.CenterCrop(input_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # 3. Medium-resolution RGB datasets
    else:
        if dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']:
            crop_size = 32
            padding = 4
        elif dataset_name == 'STL10':
            crop_size = 96
            padding = 12
        else:
            crop_size = 32
            padding = 4
            
        if augment:
            train_transform = transforms.Compose([
                to_rgb,
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return train_transform, test_transform

def load_dataset(dataset_name, data_path, train, download, transform):
    """
    Loads the specified dataset from torchvision.
    For datasets without built-in splits, returns the full dataset.
    """
    stats = get_dataset_stats(dataset_name)
    
    # Datasets without built-in train/test splits
    if not stats['has_split']:
        if dataset_name == 'Caltech101':
            return datasets.Caltech101(
                root=data_path,
                download=download,
                transform=transform,
                target_type='category'
            )
        elif dataset_name == 'Caltech256':
            return datasets.Caltech256(
                root=data_path,
                download=download,
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {dataset_name} marked as no split but not handled")
    
    # Datasets with built-in splits
    if dataset_name == 'SVHN':
        split = 'train' if train else 'test'
        return datasets.SVHN(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    elif dataset_name == 'STL10':
        split = 'train' if train else 'test'
        return datasets.STL10(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    elif dataset_name == 'ImageNet':
        imagenet_root = os.path.join(data_path, 'imagenet')
        split = 'train' if train else 'val'
        return datasets.ImageNet(
            root=imagenet_root, 
            split=split,
            transform=transform
        )
    elif dataset_name == 'TinyImageNet':
        raise NotImplementedError("TinyImageNet requires custom dataset implementation.")
    elif dataset_name == 'Food101':
        split = 'train' if train else 'test'
        return datasets.Food101(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    elif dataset_name == 'Places365':
        split = 'train-standard' if train else 'val'
        return datasets.Places365(
            root=data_path,
            split=split,
            small=True,
            download=download,
            transform=transform
        )
    elif dataset_name == 'Flowers102':
        split = 'train' if train else 'test'
        return datasets.Flowers102(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    elif dataset_name == 'OxfordIIITPet':
        split = 'trainval' if train else 'test'
        return datasets.OxfordIIITPet(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    else:
        # Standard datasets (MNIST, FashionMNIST, CIFAR10, CIFAR100, KMNIST)
        dataset_class = getattr(datasets, dataset_name)
        return dataset_class(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )

def manual_train_test_split(dataset, test_split=0.2, seed=42):
    """
    Manually splits a dataset into train and test sets.
    
    Args:
        dataset: The full dataset to split
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, test_dataset: Two Subset objects
    """
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, test_dataset

class PrecomputedFeatureDataset(Dataset):
    """
    A dataset class that loads precomputed features and labels from a .pt file.
    """
    def __init__(self, file_path, rank=0):
        if rank == 0:
            print(f"Loading precomputed data from {file_path}...")
        data = torch.load(file_path, map_location='cpu')
        self.features = data['features']
        self.labels = data['labels']
        if rank == 0:
            print(f"Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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
    
    # Check for precomputed feature usage
    use_precomputed = config['data'].get('use_precomputed_features', False)
    base_path = config['data'].get('precomputed_path', 'scratch/narjis')

    if use_precomputed:
        if config['model']['name'] != 'ResNet' or not config['model']['pretrained']:
            if rank == 0:
                raise ValueError("use_precomputed_features is True, but model is not 'ResNet' with 'pretrained=True'")
        
        embedding_dir = os.path.join(base_path, dataset_name, "precomputed_embeding")

        if rank == 0:
            print(f"Using precomputed features from: {embedding_dir}")
        
        train_file = os.path.join(embedding_dir, "train.pt")
        test_file = os.path.join(embedding_dir, "test.pt")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Precomputed files not found at {embedding_dir}. Run src/precompute.py first.")
        
        full_train_dataset = PrecomputedFeatureDataset(train_file, rank)
        test_dataset = PrecomputedFeatureDataset(test_file, rank)
        
        if use_augmentation and rank == 0:
            print("Warning: Data augmentation is disabled when using precomputed features.")

    else:
        # Load raw images
        if rank == 0:
            print(f"Loading raw image data for {dataset_name}...")
        
        stats = get_dataset_stats(dataset_name)
        train_transform, test_transform = get_transforms(dataset_name, augment=use_augmentation)
        
        # Check if dataset has built-in split
        if not stats['has_split']:
            # Dataset needs manual splitting
            if rank == 0:
                print(f"{dataset_name} does not have built-in train/test split. Creating manual split...")
                try:
                    full_dataset = load_dataset(
                        dataset_name=dataset_name,
                        data_path=data_path,
                        train=True,  # Ignored for datasets without split
                        download=True,
                        transform=None  # We'll apply transforms after splitting
                    )
                except Exception as e:
                    print(f"Error loading dataset {dataset_name}: {e}")
                    raise
            
            # Wait for rank 0 to finish downloading
            torch.distributed.barrier()
            
            # Other ranks load the downloaded data
            if rank != 0:
                full_dataset = load_dataset(
                    dataset_name=dataset_name,
                    data_path=data_path,
                    train=True,
                    download=False,
                    transform=None
                )
            
            # Split into train and test
            test_split = stats.get('test_split', 0.2)
            train_dataset_indices, test_dataset_indices = manual_train_test_split(
                full_dataset, 
                test_split=test_split, 
                seed=seed
            )
            
            # Create new datasets with appropriate transforms
            # We need to wrap these Subsets to apply different transforms
            class TransformedSubset(Dataset):
                def __init__(self, subset, transform):
                    self.subset = subset
                    self.transform = transform
                
                def __len__(self):
                    return len(self.subset)
                
                def __getitem__(self, idx):
                    img, label = self.subset[idx]
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            
            full_train_dataset = TransformedSubset(train_dataset_indices, train_transform)
            test_dataset = TransformedSubset(test_dataset_indices, test_transform)
            
            if rank == 0:
                print(f"Split {dataset_name}: {len(full_train_dataset)} train, {len(test_dataset)} test")
        
        else:
            # Dataset has built-in train/test split
            if rank == 0:
                try:
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
                except Exception as e:
                    print(f"Error loading dataset {dataset_name}: {e}")
                    if dataset_name == 'ImageNet':
                        print("ImageNet requires manual download. Please download from https://image-net.org/download.php")
                    raise
            
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
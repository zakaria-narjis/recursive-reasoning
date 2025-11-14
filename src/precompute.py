# src/precompute.py
import torch
import os
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms # Need this for the model's transform

# Import from your existing files
from model import ResNet
from dataset import get_dataset_stats, load_dataset, get_transforms

def precompute_features(config, dataset_name, device):
    """
    Precomputes and saves features for a given dataset (train and test splits).
    """
    print(f"--- Starting precomputation for {dataset_name} ---")
    
    # --- MODIFIED PATH LOGIC ---
    # Get the base path from config, default to 'scratch/narjis'
    base_path = config['data']['precomputed_path']
    
    # Construct the full dataset-specific embedding path
    # e.g., scratch/narjis/CIFAR10/precomputed embeding
    embedding_dir = os.path.join(base_path, dataset_name, 'precomputed_embeding') 
    os.makedirs(embedding_dir, exist_ok=True) # Create this specific directory
    print(f"Embeddings will be saved to: {embedding_dir}")
    # --- END OF MODIFICATION ---

    data_path = config['data']['path']

    # --- 1. Load Dataset Stats and Model ---
    try:
        stats = get_dataset_stats(dataset_name)
    except ValueError as e:
        print(f"Skipping {dataset_name}: {e}")
        return

    # We need the *test* transform to get normalized, non-augmented images
    # The model itself will handle resizing and grayscale
    _, test_transform = get_transforms(dataset_name, augment=False)

    # Load the ResNet model. 
    # We set pretrained=True to get the feature extractor.
    # num_classes/recursive_mode don't matter for this.
    print("Loading pretrained ResNet-18 model...")
    model = ResNet(
        num_classes=stats['num_classes'],
        pretrained=True,
        recursive_mode=False 
    )
    model.to(device)
    model.eval()

    # --- 2. Process 'train' and 'test' splits ---
    for split in ['train', 'test']:
        print(f"Processing {dataset_name} '{split}' split...")
        
        # --- MODIFIED PATH LOGIC ---
        # Save file will be e.g., scratch/narjis/CIFAR10/precomputed embeding/train.pt
        save_file = os.path.join(embedding_dir, f"{split}.pt")
        # --- END OF MODIFICATION ---
        
        if os.path.exists(save_file):
            print(f"File already exists, skipping: {save_file}")
            continue

        # Load the raw dataset
        raw_dataset = load_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            train=(split == 'train'),
            download=True,
            transform=test_transform # Use non-augmented transform
        )
        
        # Use a simple DataLoader
        dataloader = DataLoader(
            raw_dataset,
            batch_size=config['training']['batch_size'] * 2, # Use larger batch for speed
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_features = []
        all_labels = []

        # --- 3. Extract Features ---
        for images, labels in tqdm(dataloader, desc=f"Extracting {split}"):
            images = images.to(device)
            
            # --- This logic is copied from ResNet.get_input_embedding ---
            # Handle grayscale input (B,1,H,W) -> (B,3,H,W)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Resize if using pretrained backbone
            images = model.resize_transform(images)
            
            # Extract features (B, 512, 1, 1)
            with torch.no_grad():
                # We call feature_extractor directly
                features = model.feature_extractor(images) 
            # -----------------------------------------------------------
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        # --- 4. Save to Disk ---
        print("Concatenating features...")
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        print(f"Saving {all_features.shape[0]} features to {save_file}...")
        torch.save({
            'features': all_features, # Shape: (N, 512, 1, 1)
            'labels': all_labels      # Shape: (N,)
        }, save_file)

    print(f"--- Finished precomputation for {dataset_name} ---")

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets to precompute as requested
    datasets_to_precompute = ['CIFAR10', 'CIFAR100']
    
    # You could also get this from config if you prefer
    # datasets_to_precompute = config['data'].get('precompute_list', ['CIFAR10', 'CIFAR100'])

    for dataset_name in datasets_to_precompute:
        precompute_features(config, dataset_name, device)

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute ResNet-18 features for datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file (default: config/config.yaml)"
    )
    args = parser.parse_args()
    
    main(args.config)
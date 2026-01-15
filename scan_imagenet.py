from PIL import Image
import os
from pathlib import Path

def check_imagenet_integrity(imagenet_path):
    """Check all images in ImageNet directory"""
    corrupted = []
    
    for split in ['train', 'val']:
        split_path = Path(imagenet_path) / split
        if not split_path.exists():
            continue
            
        for img_path in split_path.rglob('*.JPEG'):
            try:
                img = Image.open(img_path)
                img.verify()  # Verify it's a valid image
                img = Image.open(img_path)  # Reopen after verify
                img.convert('RGB')  # Try the actual conversion
            except Exception as e:
                print(f"Corrupted: {img_path}")
                corrupted.append(str(img_path))
    
    return corrupted

# Run this
imagenet_path = "/scratch/narjis/imagenet"  # Update this
corrupted = check_imagenet_integrity(imagenet_path)
print(f"Found {len(corrupted)} corrupted images")

# Optionally remove them
for path in corrupted:
    os.remove(path)
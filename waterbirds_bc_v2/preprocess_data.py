"""
Data preprocessing script for Waterbirds dataset
"""

import os
import shutil
import numpy as np
from typing import Dict, List, Tuple
import json
from PIL import Image
import torch
from torchvision import transforms
from config.config import *

class WaterbirdsDataProcessor:
    """Processor for Waterbirds dataset"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.group_mapping = {
            'waterbird_water': 'WB-Water',
            'waterbird_land': 'WB-Land', 
            'landbird_land': 'LB-Land',
            'landbird_water': 'LB-Water'
        }
        
    def validate_dataset_structure(self):
        """Validate that dataset has expected structure"""
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.data_dir, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory {dir_path} not found")
        
        print("Dataset structure validated successfully")
    
    def create_group_directories(self):
        """Create output directories for each group"""
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for group in GROUP_NAMES:
                group_dir = os.path.join(split_dir, group)
                os.makedirs(group_dir, exist_ok=True)
    
    def process_split(self, split: str, distribution: Dict[str, float] = None):
        """Process a specific split of the dataset"""
        print(f"Processing {split} split...")
        
        split_dir = os.path.join(self.data_dir, split)
        output_split_dir = os.path.join(self.output_dir, split)
        
        # Get all image files
        image_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
        
        # Group images by group type
        group_files = {group: [] for group in GROUP_NAMES}
        
        for img_file in image_files:
            # Extract group information from filename
            # Assuming format: {group_type}_{bird_id}_{background}.jpg
            parts = img_file.split('_')
            if len(parts) >= 3:
                bird_type = parts[0]
                background = parts[2].split('.')[0]
                
                # Map to group name
                if bird_type == 'waterbird' and background == 'water':
                    group = 'WB-Water'
                elif bird_type == 'waterbird' and background == 'land':
                    group = 'WB-Land'
                elif bird_type == 'landbird' and background == 'land':
                    group = 'LB-Land'
                elif bird_type == 'landbird' and background == 'water':
                    group = 'LB-Water'
                else:
                    continue
                
                group_files[group].append((img_file, os.path.join(split_dir, img_file)))
        
        # Apply distribution sampling if provided
        if distribution:
            total_samples = sum(len(files) for files in group_files.values())
            target_samples = {group: int(prob * total_samples) 
                             for group, prob in distribution.items()}
            
            # Sample files
            for group in GROUP_NAMES:
                if group in distribution and group_files[group]:
                    n_samples = min(target_samples[group], len(group_files[group]))
                    sampled_files = np.random.choice(
                        len(group_files[group]), n_samples, replace=False
                    )
                    group_files[group] = [group_files[group][i] for i in sampled_files]
        
        # Copy files to group directories
        for group, files in group_files.items():
            group_dir = os.path.join(output_split_dir, group)
            print(f"  {group}: {len(files)} images")
            
            for img_file, img_path in files:
                dest_path = os.path.join(group_dir, img_file)
                shutil.copy2(img_path, dest_path)
        
        return group_files
    
    def create_metadata(self, train_distribution: Dict[str, float] = None):
        """Create metadata files for the dataset"""
        metadata = {
            'train_distribution': train_distribution,
            'group_names': GROUP_NAMES,
            'num_groups': NUM_GROUPS,
            'seeds': SEEDS
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
    
    def process_dataset(self, train_distribution: Dict[str, float] = None):
        """Process the entire dataset"""
        print("Starting Waterbirds dataset processing...")
        
        # Validate dataset structure
        self.validate_dataset_structure()
        
        # Create output directories
        self.create_group_directories()
        
        # Process each split
        for split in ['train', 'val', 'test']:
            self.process_split(split, train_distribution if split == 'train' else None)
        
        # Create metadata
        self.create_metadata(train_distribution)
        
        print("Dataset processing completed!")
    
    def verify_dataset(self):
        """Verify the processed dataset"""
        print("Verifying processed dataset...")
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            print(f"\n{split.capitalize()} split:")
            
            total_images = 0
            for group in GROUP_NAMES:
                group_dir = os.path.join(split_dir, group)
                if os.path.exists(group_dir):
                    num_images = len([f for f in os.listdir(group_dir) if f.endswith('.jpg')])
                    print(f"  {group}: {num_images} images")
                    total_images += num_images
                else:
                    print(f"  {group}: 0 images")
            
            print(f"  Total: {total_images} images")

def setup_kaggle_environment():
    """Setup environment for Kaggle execution"""
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "summary"), exist_ok=True)
    
    # Create requirements.txt
    requirements = """
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
tqdm>=4.62.0
Pillow>=8.3.0
    """
    
    with open(os.path.join(BASE_DIR, "requirements.txt"), 'w') as f:
        f.write(requirements.strip())
    
    print("Environment setup completed!")

def main():
    """Main preprocessing function"""
    print("Waterbirds Dataset Preprocessing")
    print("=" * 50)
    
    # Setup environment
    setup_kaggle_environment()
    
    # Initialize processor
    data_dir = os.path.join(BASE_DIR, "data")
    output_dir = os.path.join(BASE_DIR, "data_processed")
    
    processor = WaterbirdsDataProcessor(data_dir, output_dir)
    
    # Process dataset with training distribution
    processor.process_dataset(train_distribution=GLOBAL_TRAIN_DISTRIBUTION)
    
    # Verify dataset
    processor.verify_dataset()
    
    print("\nPreprocessing completed successfully!")
    print("You can now run the main experiment script.")

if __name__ == "__main__":
    main()
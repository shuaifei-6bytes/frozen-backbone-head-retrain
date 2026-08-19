"""
Dataset utilities for Waterbirds B/C experiment V2
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import json

class WaterbirdsDataset(Dataset):
    """Custom Waterbirds dataset with group-based sampling control"""
    
    def __init__(self, root_dir: str, split: str = "train", 
                 distribution: Dict[str, float] = None,
                 transform: Optional[transforms.Compose] = None,
                 seed: int = 42):
        """
        Args:
            root_dir: Root directory containing waterbirds dataset
            split: 'train', 'val', or 'test'
            distribution: Dictionary mapping group names to sampling probabilities
            transform: Data augmentation pipeline
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.split = split
        self.distribution = distribution or {}
        self.transform = transform or self._get_default_transform()
        self.seed = seed
        
        # Load dataset metadata
        self.samples, self.labels, self.groups = self._load_metadata()
        
        # Filter samples based on split
        self._filter_by_split()
        
        # Apply distribution sampling if provided
        if self.distribution:
            self._apply_distribution_sampling()
        
        # Create group mapping
        self.group_to_idx = {group: i for i, group in enumerate(GROUP_NAMES)}
        
    def _get_default_transform(self):
        """Get default transformation pipeline"""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_metadata(self):
        """Load sample metadata and group information"""
        # This is a placeholder - actual implementation depends on dataset structure
        # Assuming dataset has a metadata file or can be inferred from filenames
        samples = []
        labels = []
        groups = []
        
        # For now, create dummy data structure
        # In practice, you would load actual waterbirds dataset
        num_samples = 1000  # Placeholder
        
        for i in range(num_samples):
            # Simulate different groups
            group_idx = i % NUM_GROUPS
            group = GROUP_NAMES[group_idx]
            label = 0 if group in ["WB-Water", "WB-Land"] else 1
            
            samples.append(f"sample_{i}.jpg")
            labels.append(label)
            groups.append(group)
            
        return samples, labels, groups
    
    def _filter_by_split(self):
        """Filter samples based on split"""
        # This is a placeholder - actual implementation depends on dataset structure
        # For now, assume we have a way to determine split
        if self.split == "train":
            keep_indices = list(range(int(0.8 * len(self.samples))))
        elif self.split == "val":
            keep_indices = list(range(int(0.8 * len(self.samples)), 
                                   int(0.9 * len(self.samples))))
        else:  # test
            keep_indices = list(range(int(0.9 * len(self.samples)), len(self.samples)))
            
        self.samples = [self.samples[i] for i in keep_indices]
        self.labels = [self.labels[i] for i in keep_indices]
        self.groups = [self.groups[i] for i in keep_indices]
    
    def _apply_distribution_sampling(self):
        """Apply distribution-based sampling"""
        if not self.distribution:
            return
            
        # Group samples by group
        group_samples = {group: [] for group in GROUP_NAMES}
        group_labels = {group: [] for group in GROUP_NAMES}
        
        for i, group in enumerate(self.groups):
            group_samples[group].append(self.samples[i])
            group_labels[group].append(self.labels[i])
        
        # Sample according to distribution
        total_samples = sum(len(samples) for samples in group_samples.values())
        target_samples = {group: int(prob * total_samples) 
                         for group, prob in self.distribution.items()}
        
        # Apply sampling
        new_samples = []
        new_labels = []
        new_groups = []
        
        for group in GROUP_NAMES:
            if group in self.distribution and group_samples[group]:
                n_samples = min(target_samples[group], len(group_samples[group]))
                indices = np.random.RandomState(self.seed).choice(
                    len(group_samples[group]), n_samples, replace=False)
                
                for idx in indices:
                    new_samples.append(group_samples[group][idx])
                    new_labels.append(group_labels[group][idx])
                    new_groups.append(group)
        
        self.samples = new_samples
        self.labels = new_labels
        self.groups = new_groups
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.root_dir, self.samples[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get label and group
        label = self.labels[idx]
        group = self.groups[idx]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'group': group,
            'sample_id': self.samples[idx]
        }

class CounterfactualDataset(Dataset):
    """Dataset for counterfactual evaluation"""
    
    def __init__(self, root_dir: str, bird_ids: List[str], 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Root directory containing waterbirds dataset
            bird_ids: List of bird IDs to create counterfactual pairs
            transform: Data augmentation pipeline
        """
        self.root_dir = root_dir
        self.bird_ids = bird_ids
        self.transform = transform or self._get_default_transform()
        
        # Create counterfactual pairs
        self.counterfactual_pairs = self._create_counterfactual_pairs()
    
    def _get_default_transform(self):
        """Get default transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_counterfactual_pairs(self):
        """Create counterfactual image pairs for each bird"""
        pairs = []
        
        for bird_id in self.bird_ids:
            # Create pair: same bird with different backgrounds
            pair = {
                'bird_id': bird_id,
                'water_background': f"{bird_id}_water.jpg",
                'land_background': f"{bird_id}_land.jpg"
            }
            pairs.append(pair)
        
        return pairs
    
    def __len__(self):
        return len(self.counterfactual_pairs)
    
    def __getitem__(self, idx):
        pair = self.counterfactual_pairs[idx]
        
        # Load both images
        water_img = Image.open(os.path.join(self.root_dir, pair['water_background'])).convert('RGB')
        land_img = Image.open(os.path.join(self.root_dir, pair['land_background'])).convert('RGB')
        
        water_img = self.transform(water_img)
        land_img = self.transform(land_img)
        
        return {
            'bird_id': pair['bird_id'],
            'water_background': water_img,
            'land_background': land_img
        }

def create_data_loaders(data_dir: str, batch_size: int = 32, 
                       distribution: Dict[str, float] = None,
                       split: str = "train") -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Create datasets
    train_dataset = WaterbirdsDataset(
        root_dir=data_dir,
        split="train",
        distribution=distribution,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    val_dataset = WaterbirdsDataset(
        root_dir=data_dir,
        split="val",
        distribution=distribution,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_counterfactual_loader(data_dir: str, bird_ids: List[str], 
                               batch_size: int = 32) -> DataLoader:
    """Create counterfactual evaluation data loader"""
    
    dataset = CounterfactualDataset(
        root_dir=data_dir,
        bird_ids=bird_ids,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
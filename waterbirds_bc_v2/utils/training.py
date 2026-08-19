"""
Training utilities for Waterbirds B/C experiment V2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from tqdm import tqdm
from utils.model import ResNetWithHead, save_model, load_model
from utils.dataset import create_data_loaders

class Trainer:
    """Trainer class for federated learning and head retraining"""
    
    def __init__(self, model: ResNetWithHead, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 0.001,
                 device: torch.device = DEVICE):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update scheduler
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(self, epochs: int, save_path: Optional[str] = None) -> List[Dict[str, float]]:
        """Train for multiple epochs"""
        history = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            }
            
            history.append(epoch_metrics)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save checkpoint if path provided
            if save_path:
                checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
                save_model(self.model, checkpoint_path)
        
        # Save final model
        if save_path:
            save_model(self.model, save_path)
        
        return history

class FederatedTrainer:
    """Federated learning trainer"""
    
    def __init__(self, num_clients: int, 
                 train_loaders: List[DataLoader],
                 val_loaders: List[DataLoader],
                 learning_rate: float = 0.001,
                 device: torch.device = DEVICE):
        self.num_clients = num_clients
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Create client models
        self.client_models = []
        for _ in range(num_clients):
            model = ResNetWithHead(num_classes=2, freeze_backbone=True)
            self.client_models.append(model.to(device))
        
        # Global model
        self.global_model = ResNetWithHead(num_classes=2, freeze_backbone=True).to(device)
        
        # Setup optimizer for global model
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.global_model.parameters()),
            lr=learning_rate
        )
    
    def client_update(self, client_idx: int, epochs: int = 1) -> Dict[str, float]:
        """Update a single client model"""
        client_model = self.client_models[client_idx]
        train_loader = self.train_loaders[client_idx]
        
        # Copy global weights to client
        client_model.load_state_dict(self.global_model.state_dict())
        
        # Setup client optimizer
        client_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, client_model.parameters()),
            lr=self.optimizer.param_groups[0]['lr']
        )
        
        # Train client
        client_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                client_optimizer.zero_grad()
                outputs = client_model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                client_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / (len(train_loader) * epochs),
            'accuracy': 100. * correct / (total * epochs)
        }
    
    def federated_average(self):
        """Perform federated averaging"""
        # Initialize global model with zero gradients
        for param in self.global_model.parameters():
            param.data.zero_()
        
        # Average client weights
        for client_model in self.client_models:
            for global_param, client_param in zip(
                self.global_model.parameters(), client_model.parameters()
            ):
                if client_param.requires_grad:
                    global_param.data += client_param.data / self.num_clients
    
    def train_round(self) -> Dict[str, float]:
        """Train one federated round"""
        client_metrics = []
        
        # Update each client
        for client_idx in range(self.num_clients):
            metrics = self.client_update(client_idx, epochs=LOCAL_EPOCHS)
            client_metrics.append(metrics)
        
        # Federated averaging
        self.federated_average()
        
        # Calculate average metrics
        avg_loss = np.mean([m['loss'] for m in client_metrics])
        avg_acc = np.mean([m['accuracy'] for m in client_metrics])
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'client_metrics': client_metrics
        }
    
    def train(self, rounds: int) -> List[Dict[str, float]]:
        """Train for multiple rounds"""
        history = []
        
        for round_num in range(rounds):
            print(f"\nFederated Round {round_num + 1}/{rounds}")
            
            # Train round
            round_metrics = self.train_round()
            
            history.append({
                'round': round_num + 1,
                **round_metrics
            })
            
            print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}, "
                  f"Avg Acc: {round_metrics['avg_accuracy']:.2f}%")
        
        return history
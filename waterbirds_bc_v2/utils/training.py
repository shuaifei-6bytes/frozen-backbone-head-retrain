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
from datetime import datetime
from utils.model import ResNetWithHead, save_model, load_model
from utils.dataset import create_data_loaders
import config.config

# tqdm 在 nohup 日志/Kaggle 后台输出里会逐 batch 刷屏（每行一个 \r 进度条），
# 淹没关键训练日志；改用轻量进度打印，每个 epoch 汇总一次
from tqdm import tqdm

def _tqdm(iterable, desc="", total=None):
    """静默包装器：直接透传迭代，不产生逐 batch 进度条输出"""
    return iterable

# Get device from config
DEVICE = config.config.DEVICE
LOCAL_EPOCHS = config.config.LOCAL_EPOCHS

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
        # 冻结 backbone 时保持其 eval：防止仅训 head 时 BN running 统计被刷新/漂移
        self.model.backbone.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(_tqdm(self.train_loader, desc="Training")):
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
            for batch in _tqdm(self.val_loader, desc="Validating"):
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

            # 只保留最终模型，不再每个 epoch 存 checkpoint：
            # 完整模型约 200MB/个，B/C 共 40 个/seed，4 seed 约 32GB，必超 Kaggle 20GB 磁盘限额
            # 训练历史已完整记录在 history json，需要中间权重可按需恢复
        
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

        # 打印各客户端样本量，便于确认 IID 数据划分已生效
        for i, tl in enumerate(self.train_loaders):
            print(f"  [Fed] client {i + 1}/{num_clients} 样本量: {len(tl.dataset)}", flush=True)
    
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
        client_model.backbone.eval()  # 冻结 BN：head 训练时 backbone 保持 eval
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            e_loss = 0.0
            e_correct = 0
            e_total = 0
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                client_optimizer.zero_grad()
                outputs = client_model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                client_optimizer.step()

                total_loss += loss.item()
                e_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                e_total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                e_correct += predicted.eq(labels).sum().item()

            # 每个 epoch 实时汇报：让 run.log 全程持续推进，绝不长时间静默
            print(f"      [Fed] client {client_idx + 1} epoch {epoch + 1}/{epochs} "
                  f"| loss={e_loss / max(len(train_loader), 1):.4f} "
                  f"acc={100.0 * e_correct / max(e_total, 1):.2f}%", flush=True)

        return {
            'loss': total_loss / (len(train_loader) * epochs),
            'accuracy': 100.0 * correct / max(total, 1)
        }
    
    def federated_average(self):
        """Perform federated averaging

        只对可训练参数（head）做清零+平均；冻结的 backbone requires_grad=False，
        绝不能碰——否则每轮结束 backbone 会被清零且不还原，模型特征被彻底破坏。
        """
        # Initialize global head params with zeros
        for param in self.global_model.parameters():
            if param.requires_grad:
                param.data.zero_()

        # Average client head weights
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
            print(f"    client {client_idx + 1}/{self.num_clients} 完成 | "
                  f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.2f}%", flush=True)
        
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
        """Train for multiple rounds (含时间统计，便于判断进度与是否卡住)"""
        history = []
        t0 = time.time()

        for round_num in range(rounds):
            elapsed = (time.time() - t0) / 60.0
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"第 {round_num + 1}/{rounds} 轮 | 已用 {elapsed:.1f} 分钟 | 开始...", flush=True)

            # Train round
            round_metrics = self.train_round()

            history.append({
                'round': round_num + 1,
                **round_metrics
            })

            elapsed2 = (time.time() - t0) / 60.0
            eta = elapsed2 / (round_num + 1) * (rounds - round_num - 1)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"第 {round_num + 1}/{rounds} 轮完成 | Loss: {round_metrics['avg_loss']:.4f} "
                  f"Acc: {round_metrics['avg_accuracy']:.2f}% | 已用 {elapsed2:.1f} 分, 预计剩 {eta:.1f} 分",
                  flush=True)

        print(f"\n联邦训练结束，总耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        return history
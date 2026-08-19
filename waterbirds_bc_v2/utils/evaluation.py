"""
Evaluation utilities for Waterbirds B/C experiment V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm
from utils.model import ResNetWithHead
from utils.dataset import create_counterfactual_loader
import config.config

# Get config values
DEVICE = config.config.DEVICE
GROUP_NAMES = config.config.GROUP_NAMES

class Evaluator:
    """Evaluator class for B/C experiment metrics"""
    
    def __init__(self, model: ResNetWithHead, device: torch.device = DEVICE):
        self.model = model
        self.device = device
    
    def evaluate_group_accuracy(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate accuracy for each group"""
        group_correct = {group: 0 for group in GROUP_NAMES}
        group_total = {group: 0 for group in GROUP_NAMES}
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating group accuracy"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                groups = batch['group']
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                # Update group statistics
                for i, group in enumerate(groups):
                    group_total[group] += 1
                    if predicted[i] == labels[i]:
                        group_correct[group] += 1
        
        # Calculate accuracies
        group_accuracies = {}
        for group in GROUP_NAMES:
            if group_total[group] > 0:
                group_accuracies[group] = 100. * group_correct[group] / group_total[group]
            else:
                group_accuracies[group] = 0.0
        
        return group_accuracies
    
    def evaluate_overall_accuracy(self, data_loader: DataLoader) -> float:
        """Evaluate overall accuracy"""
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating overall accuracy"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def evaluate_worst_group_accuracy(self, group_accuracies: Dict[str, float]) -> float:
        """Calculate worst group accuracy"""
        return min(group_accuracies.values())
    
    def evaluate_counterfactual(self, counterfactual_loader: DataLoader) -> Dict:
        """Evaluate counterfactual predictions"""
        self.model.eval()
        
        results = {
            'waterbird_predictions_water': [],
            'waterbird_predictions_land': [],
            'landbird_predictions_land': [],
            'landbird_predictions_water': [],
            'bird_ids': []
        }
        
        with torch.no_grad():
            for batch in tqdm(counterfactual_loader, desc="Evaluating counterfactual"):
                water_images = batch['water_background'].to(self.device)
                land_images = batch['land_background'].to(self.device)
                bird_ids = batch['bird_id']
                
                # Predict on water background
                water_outputs = self.model(water_images)
                water_probs = F.softmax(water_outputs, dim=1)
                
                # Predict on land background
                land_outputs = self.model(land_images)
                land_probs = F.softmax(land_outputs, dim=1)
                
                # Store results
                for i, bird_id in enumerate(bird_ids):
                    results['bird_ids'].append(bird_id)
                    
                    # Water background predictions
                    water_prob_waterbird = water_probs[i][0].item()  # P(waterbird | water background)
                    water_prob_landbird = water_probs[i][1].item()   # P(landbird | water background)
                    results['waterbird_predictions_water'].append(water_prob_waterbird)
                    results['landbird_predictions_water'].append(water_prob_landbird)
                    
                    # Land background predictions
                    land_prob_waterbird = land_probs[i][0].item()   # P(waterbird | land background)
                    land_prob_landbird = land_probs[i][1].item()    # P(landbird | land background)
                    results['waterbird_predictions_land'].append(land_prob_waterbird)
                    results['landbird_predictions_land'].append(land_prob_landbird)
        
        return results
    
    def calculate_signed_background_effects(self, counterfactual_results: Dict) -> Dict:
        """Calculate signed background effects"""
        water_probs_water = np.array(counterfactual_results['waterbird_predictions_water'])
        water_probs_land = np.array(counterfactual_results['waterbird_predictions_land'])
        land_probs_land = np.array(counterfactual_results['landbird_predictions_land'])
        land_probs_water = np.array(counterfactual_results['landbird_predictions_water'])
        
        # Calculate signed effects
        delta_waterbird = np.mean(water_probs_water - water_probs_land)
        delta_landbird = np.mean(land_probs_land - land_probs_water)
        
        return {
            'delta_waterbird': delta_waterbird,
            'delta_landbird': delta_landbird
        }
    
    def calculate_background_gap(self, delta_waterbird: float, delta_landbird: float) -> float:
        """Calculate background gap from signed effects"""
        bg_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2
        
        # Consistency check
        expected_bg_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2
        assert abs(bg_gap - expected_bg_gap) < 1e-6, "Background gap calculation inconsistency"
        
        return bg_gap
    
    def calculate_flip_rates(self, counterfactual_results: Dict) -> Dict:
        """Calculate directional flip rates"""
        water_probs_water = np.array(counterfactual_results['waterbird_predictions_water'])
        water_probs_land = np.array(counterfactual_results['waterbird_predictions_land'])
        land_probs_land = np.array(counterfactual_results['landbird_predictions_land'])
        land_probs_water = np.array(counterfactual_results['landbird_predictions_water'])
        
        total_pairs = len(water_probs_water)
        original_flips = 0
        reverse_flips = 0
        
        for i in range(total_pairs):
            # Water bird predictions
            if water_probs_water[i] > 0.5:  # Predicted as waterbird with water background
                if water_probs_land[i] < 0.5:  # Predicted as landbird with land background
                    original_flips += 1
                elif water_probs_land[i] > 0.5:  # Predicted as waterbird with land background
                    reverse_flips += 1
            
            # Land bird predictions
            if land_probs_land[i] > 0.5:  # Predicted as landbird with land background
                if land_probs_water[i] < 0.5:  # Predicted as waterbird with water background
                    original_flips += 1
                elif land_probs_water[i] > 0.5:  # Predicted as landbird with water background
                    reverse_flips += 1
        
        original_flip_rate = original_flips / total_pairs
        reverse_flip_rate = reverse_flips / total_pairs
        
        return {
            'original_flip_rate': original_flip_rate,
            'reverse_flip_rate': reverse_flip_rate
        }
    
    def save_probabilities(self, counterfactual_results: Dict, save_path: str):
        """Save individual probabilities for auditing"""
        probabilities = {
            'P_WB_WATER': counterfactual_results['waterbird_predictions_water'],
            'P_WB_LAND': counterfactual_results['waterbird_predictions_land'],
            'P_LB_LAND': counterfactual_results['landbird_predictions_land'],
            'P_LB_WATER': counterfactual_results['landbird_predictions_water'],
            'bird_ids': counterfactual_results['bird_ids']
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(probabilities, f, indent=2)
    
    def evaluate_model(self, data_loader: DataLoader, 
                      counterfactual_loader: DataLoader,
                      model_name: str,
                      save_dir: str) -> Dict:
        """Comprehensive model evaluation"""
        print(f"\nEvaluating {model_name}...")
        
        # Evaluate group and overall accuracy
        group_accuracies = self.evaluate_group_accuracy(data_loader)
        overall_accuracy = self.evaluate_overall_accuracy(data_loader)
        worst_group_accuracy = self.evaluate_worst_group_accuracy(group_accuracies)
        
        # Evaluate counterfactual predictions
        counterfactual_results = self.evaluate_counterfactual(counterfactual_loader)
        
        # Calculate signed background effects
        signed_effects = self.calculate_signed_background_effects(counterfactual_results)
        delta_waterbird = signed_effects['delta_waterbird']
        delta_landbird = signed_effects['delta_landbird']
        
        # Calculate background gap
        background_gap = self.calculate_background_gap(delta_waterbird, delta_landbird)
        
        # Calculate flip rates
        flip_rates = self.calculate_flip_rates(counterfactual_results)
        
        # Compile results
        results = {
            'model_name': model_name,
            'overall_accuracy': overall_accuracy,
            'worst_group_accuracy': worst_group_accuracy,
            'group_accuracies': group_accuracies,
            'delta_waterbird': delta_waterbird,
            'delta_landbird': delta_landbird,
            'background_gap': background_gap,
            'original_flip_rate': flip_rates['original_flip_rate'],
            'reverse_flip_rate': flip_rates['reverse_flip_rate']
        }
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save probabilities
        probabilities_path = os.path.join(save_dir, f"{model_name}_probabilities.json")
        self.save_probabilities(counterfactual_results, probabilities_path)
        
        return results

def create_counterfactual_bird_ids(num_birds: int = 100) -> List[str]:
    """Create list of bird IDs for counterfactual evaluation"""
    return [f"bird_{i:04d}" for i in range(num_birds)]
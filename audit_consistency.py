#!/usr/bin/env python3
"""
Audit script: Verify consistency between Background Gap and Signed Background Effect

Goal: Verify if compute_background_gap_real() and compute_signed_background_effect()
satisfy BG_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "第一次实验，仅一个seed" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "验证方法C是否产生反向推力" / "src"))

# Import modules
import evaluate as eval_module
import diagnostic as diag_module

def compute_manual_bg_gap(delta_waterbird, delta_landbird):
    """Manual calculation of Background Gap"""
    return (abs(delta_waterbird) + abs(delta_landbird)) / 2

def print_four_probabilities(probs_ww, preds_ww, y_ww, probs_wl, preds_wl, y_wl,
                           probs_ll, preds_ll, y_ll, probs_lw, preds_lw, y_lw):
    """Print four underlying probabilities"""
    print("=" * 60)
    print("🔍 Four underlying probability values")
    print("=" * 60)
    
    # Ensure data exists
    if len(probs_ww) > 0:
        p_wb_water = probs_ww[:, 0].mean().item()
        print(f"P_WB_WATER = P(waterbird|waterbird主体+water背景) = {p_wb_water:.6f}")
    else:
        p_wb_water = 0.0
        print("P_WB_WATER = No data available")
    
    if len(probs_wl) > 0:
        p_wb_land = probs_wl[:, 0].mean().item()
        print(f"P_WB_LAND  = P(waterbird|waterbird主体+land背景) = {p_wb_land:.6f}")
    else:
        p_wb_land = 0.0
        print("P_WB_LAND  = No data available")
    
    if len(probs_ll) > 0:
        p_lb_land = probs_ll[:, 1].mean().item()
        print(f"P_LB_LAND  = P(landbird|landbird主体+land背景) = {p_lb_land:.6f}")
    else:
        p_lb_land = 0.0
        print("P_LB_LAND  = No data available")
    
    if len(probs_lw) > 0:
        p_lb_water = probs_lw[:, 1].mean().item()
        print(f"P_LB_WATER = P(landbird|landbird主体+water背景) = {p_lb_water:.6f}")
    else:
        p_lb_water = 0.0
        print("P_LB_WATER = No data available")
    
    print("\n" + "=" * 60)
    print("📐 Manual calculation")
    print("=" * 60)
    
    delta_waterbird = p_wb_water - p_wb_land
    delta_landbird = p_lb_land - p_lb_water
    manual_bg_gap = compute_manual_bg_gap(delta_waterbird, delta_landbird)
    
    print(f"delta_waterbird = P_WB_WATER - P_WB_LAND = {delta_waterbird:.6f}")
    print(f"delta_landbird  = P_LB_LAND - P_LB_WATER = {delta_landbird:.6f}")
    print(f"manual_bg_gap   = (|{delta_waterbird:.6f}| + |{delta_landbird:.6f}|) / 2 = {manual_bg_gap:.6f}")
    
    return delta_waterbird, delta_landbird, manual_bg_gap

def audit_model_consistency(model, group_loaders, device='cpu'):
    """Audit model consistency"""
    print(f"\n{'='*80}")
    print(f"🔍 Audit model consistency")
    print(f"{'='*80}")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Call both functions with same group loaders
    print("\n📊 Calling existing functions...")
    
    # Old function
    old_bg_gap = eval_module.compute_background_gap_real(model, group_loaders, device)
    print(f"old_function_bg_gap = compute_background_gap_real() = {old_bg_gap:.6f}")
    
    # New function
    delta_wb, delta_lb = diag_module.compute_signed_background_effect(model, group_loaders, device)
    print(f"new_function_delta_waterbird = {delta_wb:.6f}")
    print(f"new_function_delta_landbird  = {delta_lb:.6f}")
    
    # Manual calculation
    manual_bg_gap = compute_manual_bg_gap(delta_wb, delta_lb)
    print(f"manual_bg_gap = (|{delta_wb:.6f}| + |{delta_lb:.6f}|) / 2 = {manual_bg_gap:.6f}")
    
    # Check consistency
    error = abs(old_bg_gap - manual_bg_gap)
    print(f"\n🔬 Consistency check:")
    print(f"Absolute error = |{old_bg_gap:.6f} - {manual_bg_gap:.6f}| = {error:.8f}")
    
    # Print four underlying probabilities
    # Need to get data from group_loaders to recalculate
    probs_ww, _, y_ww = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, y_wl = collect_probs_preds(model, group_loaders['waterbird_land'], device)
    probs_ll, _, y_ll = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, y_lw = collect_probs_preds(model, group_loaders['landbird_water'], device)
    
    delta_wb_manual, delta_lb_manual, manual_bg_gap_manual = print_four_probabilities(
        probs_ww, None, y_ww, probs_wl, None, y_wl,
        probs_ll, None, y_ll, probs_lw, None, y_lw
    )
    
    # Final consistency check
    final_error = abs(old_bg_gap - manual_bg_gap_manual)
    print(f"\n🎯 Final consistency check:")
    print(f"Old function vs manual calculation: |{old_bg_gap:.6f} - {manual_bg_gap_manual:.6f}| = {final_error:.8f}")
    
    # Determine if consistent
    tolerance = 1e-5
    is_consistent = final_error < tolerance
    
    print(f"\n{'✅ Consistent' if is_consistent else '❌ Inconsistent'} (tolerance: {tolerance})")
    
    return {
        'old_bg_gap': old_bg_gap,
        'delta_waterbird': delta_wb,
        'delta_landbird': delta_lb,
        'manual_bg_gap': manual_bg_gap_manual,
        'error': final_error,
        'is_consistent': is_consistent,
        'four_probabilities': {
            'P_WB_WATER': probs_ww[:, 0].mean().item() if len(probs_ww) > 0 else 0.0,
            'P_WB_LAND': probs_wl[:, 0].mean().item() if len(probs_wl) > 0 else 0.0,
            'P_LB_LAND': probs_ll[:, 1].mean().item() if len(probs_ll) > 0 else 0.0,
            'P_LB_WATER': probs_lw[:, 1].mean().item() if len(probs_lw) > 0 else 0.0,
        }
    }

def collect_probs_preds(model, loader, device='cpu'):
    """Collect softmax probabilities and predictions for given loader"""
    model.eval()
    
    probs_list = []
    preds_list = []
    labels_list = []
    
    for images, labels, _ in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        
        probs_list.append(probs.cpu())
        preds_list.append(preds.cpu())
        labels_list.append(labels)
    
    if not probs_list:
        return torch.empty(0, 2), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
    return torch.cat(probs_list, 0), torch.cat(preds_list, 0), torch.cat(labels_list, 0)

def audit_group_loaders(group_loaders):
    """Audit group loaders consistency"""
    print(f"\n{'='*80}")
    print(f"🔍 Audit Group Loaders")
    print(f"{'='*80}")
    
    for group_name, loader in group_loaders.items():
        print(f"\n📁 Group: {group_name}")
        
        # Count samples
        total_samples = 0
        label_counts = {0: 0, 1: 0}  # waterbird:0, landbird:1
        
        for batch_idx, (images, labels, places) in enumerate(loader):
            total_samples += labels.size(0)
            for label in labels:
                label_counts[label.item()] += 1
            
            if batch_idx >= 2:  # Only show first few batches
                break
        
        print(f"  Total samples: {total_samples}")
        print(f"  Waterbird samples (label=0): {label_counts[0]}")
        print(f"  Landbird samples (label=1): {label_counts[1]}")
        
        # Infer background from group name
        if 'waterbird' in group_name:
            bg_type = "water background" if 'water' in group_name else "land background"
            print(f"  Subject type: waterbird, background: {bg_type}")
        else:
            bg_type = "land background" if 'land' in group_name else "water background"
            print(f"  Subject type: landbird, background: {bg_type}")

if __name__ == "__main__":
    print("🔍 Starting audit of Background Gap vs Signed Background Effect consistency")
    
    # Need to load actual model and data
    # Since no checkpoint found, create a test framework
    print("⚠️  No model checkpoint found, need to run experiments first")
    print("Please run experiment scripts to generate model files, then execute this audit")
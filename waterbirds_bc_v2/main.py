"""
Main experiment script for Waterbirds B/C experiment V2
"""

import os
import torch
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from config.config import *
from utils.model import ResNetWithHead, create_model, save_model
from utils.dataset import create_data_loaders, create_counterfactual_loader
from utils.training import Trainer, FederatedTrainer
from utils.evaluation import Evaluator, create_counterfactual_bird_ids

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_directory(seed: int) -> str:
    """Create output directory for a specific seed"""
    seed_dir = os.path.join(BASE_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    return seed_dir

def run_federated_training(seed: int, seed_dir: str) -> str:
    """Run federated training for a specific seed"""
    print(f"\n{'='*50}")
    print(f"Running federated training for seed {seed}")
    print(f"{'='*50}")
    
    set_seed(seed)
    
    # Create data loaders with global training distribution
    train_loader, val_loader = create_data_loaders(
        data_dir=os.path.join(BASE_DIR, "data"),
        batch_size=BATCH_SIZE,
        distribution=GLOBAL_TRAIN_DISTRIBUTION,
        split="train"
    )
    
    # Create federated trainer
    federated_trainer = FederatedTrainer(
        num_clients=NUM_CLIENTS,
        train_loaders=[train_loader] * NUM_CLIENTS,  # Simplified: same data for all clients
        val_loaders=[val_loader] * NUM_CLIENTS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Train federated model
    history = federated_trainer.train(rounds=GLOBAL_ROUNDS)
    
    # Save global model
    global_model_path = os.path.join(seed_dir, "M_global.pt")
    save_model(federated_trainer.global_model, global_model_path)
    
    # Save training history
    history_path = os.path.join(seed_dir, "federated_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Federated training completed. Model saved to {global_model_path}")
    return global_model_path

def save_head_initialization(model: ResNetWithHead, save_path: str):
    """Save head initialization parameters"""
    model.save_head_initialization(save_path)
    print(f"Head initialization saved to {save_path}")

def run_head_retraining(seed: int, global_model_path: str, seed_dir: str, 
                       retraining_type: str, distribution: Dict[str, float]) -> str:
    """Run head retraining for B or C"""
    print(f"\n{'='*50}")
    print(f"Running {retraining_type} head retraining for seed {seed}")
    print(f"{'='*50}")
    
    set_seed(seed)
    
    # Create model with frozen backbone
    model = create_model(num_classes=2, freeze_backbone=True)
    
    # Load backbone from global model
    model.load_backbone_from_checkpoint(global_model_path)
    
    # Load head initialization
    head_init_path = os.path.join(seed_dir, "head_init.pt")
    if os.path.exists(head_init_path):
        model.load_head_initialization(head_init_path)
    else:
        # Save head initialization if not exists
        save_head_initialization(model, head_init_path)
    
    # Create data loaders with retraining distribution
    train_loader, val_loader = create_data_loaders(
        data_dir=os.path.join(BASE_DIR, "data"),
        batch_size=HEAD_RETRAIN_BATCH_SIZE,
        distribution=distribution,
        split="train"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=HEAD_RETRAIN_LR,
        device=DEVICE
    )
    
    # Train head
    model_path = os.path.join(seed_dir, f"{retraining_type}.pt")
    history = trainer.train(
        epochs=HEAD_RETRAIN_EPOCHS,
        save_path=model_path
    )
    
    # Save training history
    history_path = os.path.join(seed_dir, f"{retraining_type}_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"{retraining_type} head retraining completed. Model saved to {model_path}")
    return model_path

def evaluate_models(seed: int, seed_dir: str, 
                   global_model_path: str, 
                   b_model_path: str, 
                   c_model_path: str):
    """Evaluate all models for a specific seed"""
    print(f"\n{'='*50}")
    print(f"Evaluating models for seed {seed}")
    print(f"{'='*50}")
    
    set_seed(seed)
    
    # Create counterfactual loader
    bird_ids = create_counterfactual_bird_ids(num_birds=100)
    counterfactual_loader = create_counterfactual_loader(
        data_dir=os.path.join(BASE_DIR, "data"),
        bird_ids=bird_ids,
        batch_size=EVAL_BATCH_SIZE
    )
    
    # Create evaluation data loader
    _, eval_loader = create_data_loaders(
        data_dir=os.path.join(BASE_DIR, "data"),
        batch_size=EVAL_BATCH_SIZE,
        distribution=None,  # Use full distribution for evaluation
        split="val"
    )
    
    # Models to evaluate
    models = [
        ("M_global", global_model_path),
        ("B", b_model_path),
        ("C", c_model_path)
    ]
    
    results = {}
    
    for model_name, model_path in models:
        # Load model
        model = create_model(num_classes=2, freeze_backbone=True)
        load_model(model, model_path, load_head=True)
        
        # Create evaluator
        evaluator = Evaluator(model, device=DEVICE)
        
        # Evaluate model
        model_results = evaluator.evaluate_model(
            data_loader=eval_loader,
            counterfactual_loader=counterfactual_loader,
            model_name=model_name,
            save_dir=seed_dir
        )
        
        results[model_name] = model_results
        
        print(f"\n{model_name} Results:")
        print(f"Overall Accuracy: {model_results['overall_accuracy']:.2f}%")
        print(f"Worst Group Accuracy: {model_results['worst_group_accuracy']:.2f}%")
        print(f"Delta Waterbird: {model_results['delta_waterbird']:.4f}")
        print(f"Delta Landbird: {model_results['delta_landbird']:.4f}")
        print(f"Background Gap: {model_results['background_gap']:.4f}")
        print(f"Original Flip Rate: {model_results['original_flip_rate']:.4f}")
        print(f"Reverse Flip Rate: {model_results['reverse_flip_rate']:.4f}")
        print(f"Group Accuracies: {model_results['group_accuracies']}")
    
    # Save combined results
    results_path = os.path.join(seed_dir, "metrics.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_single_seed_experiment(seed: int):
    """Run complete experiment for a single seed"""
    print(f"\n{'='*60}")
    print(f"Starting experiment for seed {seed}")
    print(f"{'='*60}")
    
    # Create output directory
    seed_dir = create_output_directory(seed)
    
    # Step 1: Federated training
    global_model_path = run_federated_training(seed, seed_dir)
    
    # Step 2: Head retraining
    b_model_path = run_head_retraining(seed, global_model_path, seed_dir, "B", B_DISTRIBUTION)
    c_model_path = run_head_retraining(seed, global_model_path, seed_dir, "C", C_DISTRIBUTION)
    
    # Step 3: Evaluation
    results = evaluate_models(seed, seed_dir, global_model_path, b_model_path, c_model_path)
    
    return results

def summarize_results(all_results: Dict[int, Dict]):
    """Summarize results across all seeds"""
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    
    # Extract metrics for each model
    models = ["M_global", "B", "C"]
    metrics = ["overall_accuracy", "worst_group_accuracy", "delta_waterbird", 
               "delta_landbird", "background_gap", "original_flip_rate", "reverse_flip_rate"]
    
    summary = {}
    
    for model in models:
        model_results = []
        for seed in SEEDS:
            if seed in all_results and model in all_results[seed]:
                model_results.append(all_results[seed][model])
        
        if model_results:
            summary[model] = {}
            for metric in metrics:
                values = [r[metric] for r in model_results]
                summary[model][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    # Print summary
    print("\nMean ± Standard Deviation:")
    print(f"{'Model':<12} {'Metric':<25} {'Mean':<10} {'Std':<10}")
    print("-" * 60)
    
    for model in models:
        for metric in metrics:
            if metric in summary[model]:
                mean_val = summary[model][metric]['mean']
                std_val = summary[model][metric]['std']
                print(f"{model:<12} {metric:<25} {mean_val:<10.4f} {std_val:<10.4f}")
    
    # Save summary
    summary_path = os.path.join(SUMMARY_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison B vs C
    print(f"\n{'='*60}")
    print("B vs C COMPARISON")
    print(f"{'='*60}")
    
    b_c_metrics = ["background_gap", "original_flip_rate", "reverse_flip_rate"]
    
    print(f"{'Metric':<25} {'B Mean ± Std':<20} {'C Mean ± Std':<20} {'B < C':<10}")
    print("-" * 75)
    
    for metric in b_c_metrics:
        b_mean = summary['B'][metric]['mean']
        b_std = summary['B'][metric]['std']
        c_mean = summary['C'][metric]['mean']
        c_std = summary['C'][metric]['std']
        b_less_c = b_mean < c_mean
        
        print(f"{metric:<25} {b_mean:.4f} ± {b_std:.4f}     {c_mean:.4f} ± {c_std:.4f}     {b_less_c}")
    
    return summary

def main():
    """Main experiment runner"""
    print(f"Waterbirds B/C Experiment V2")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    
    # Check if data directory exists
    data_dir = os.path.join(BASE_DIR, "data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found")
        print("Please ensure waterbirds dataset is available")
        return
    
    # Run experiments for each seed
    all_results = {}
    
    for seed in SEEDS:
        try:
            results = run_single_seed_experiment(seed)
            all_results[seed] = results
        except Exception as e:
            print(f"Error running experiment for seed {seed}: {e}")
            continue
    
    # Summarize results
    if all_results:
        summary = summarize_results(all_results)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to {SUMMARY_DIR}")
    else:
        print("No results generated. Check for errors.")

if __name__ == "__main__":
    main()
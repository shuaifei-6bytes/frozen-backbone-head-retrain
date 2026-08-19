"""
Kaggle Setup Script
This script helps setup the environment for running the experiment on Kaggle.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    # Install basic requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install additional packages that might be needed
    additional_packages = [
        "scikit-learn",
        "matplotlib", 
        "seaborn",
        "jupyter"
    ]
    
    for package in additional_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("Requirements installed successfully!")

def setup_dataset():
    """Setup dataset directory structure"""
    print("Setting up dataset directory...")
    
    # Create necessary directories
    dirs_to_create = [
        "data",
        "checkpoints", 
        "summary",
        "logs"
    ]
    
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    print("Directory structure created!")

def verify_setup():
    """Verify the setup"""
    print("Verifying setup...")
    
    # Check if CUDA is available
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check required packages
    required_packages = [
        "torch", "torchvision", "numpy", "tqdm", "PIL"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
    
    print("Setup verification completed!")

def create_run_script():
    """Create a simple run script"""
    run_script = """#!/bin/bash
# Run the Waterbirds B/C experiment

echo "Starting Waterbirds B/C Experiment V2..."
echo "Time: $(date)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the main experiment
python main.py

echo "Experiment completed at $(date)"
"""
    
    with open("run_experiment.sh", "w") as f:
        f.write(run_script)
    
    # Make it executable
    os.chmod("run_experiment.sh", 0o755)
    
    print("Run script created: run_experiment.sh")

def main():
    """Main setup function"""
    print("Waterbirds B/C Experiment V2 - Kaggle Setup")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Setup directory structure
    setup_dataset()
    
    # Verify setup
    verify_setup()
    
    # Create run script
    create_run_script()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Upload the Waterbirds dataset to /kaggle/input/waterbirds-dataset/")
    print("2. Run the experiment: python kaggle_notebook.py")
    print("3. Or run the script: ./run_experiment.sh")

if __name__ == "__main__":
    main()
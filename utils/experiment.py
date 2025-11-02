"""Experiment management tools"""
import os
from datetime import datetime
from pathlib import Path


def setup_experiment_dir(exp_name=None):
    """Set up experiment directory structure"""
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path("outputs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    sub_dirs = {
        'checkpoint_dir': exp_dir / "checkpoints",
        'best_model_dir': exp_dir / "best_model",
        'curves_dir': exp_dir / "training_curves",
        'inference_dir': exp_dir / "inference_results",
        'log_file': exp_dir / "log.txt"
    }
    
    for dir_path in sub_dirs.values():
        if str(dir_path).endswith('.txt'):
            continue  # log_file is a file not a directory
        dir_path.mkdir(exist_ok=True)
    
    return exp_name, exp_dir, sub_dirs


def print_experiment_info(exp_name, exp_dir, sub_dirs):
    """Print experiment information"""
    print(f"\n=== Experiment Setup ===")
    print(f"Experiment name: {exp_name}")
    print(f"Experiment directory: {exp_dir}")
    print(f"  Log file: {sub_dirs['log_file']}")
    print(f"  Checkpoints: {sub_dirs['checkpoint_dir']}")
    print(f"  Best model: {sub_dirs['best_model_dir']}")
    print(f"  Training curves: {sub_dirs['curves_dir']}")
    print(f"  Inference results: {sub_dirs['inference_dir']}")
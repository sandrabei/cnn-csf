# 3D Medical Image Processing Tool - Project Summary

## Project Overview
The project is divided into two main parts:
1. **Offline Data Processing**: 3D medical image data parsing, visualization slice generation and annotation application
2. **Model Training**: Training U-Net model for point detection based on processed data

Supports batch processing of all 0100* format datasets and provides complete training pipeline.

## Project Structure
```
Project Root/
├── raw_data/               # Input data directory (520 datasets)
│   ├── 01000165/          # Example dataset
│   │   ├── 01000165_V01csf.txt    # CSF annotation file (0/1 labels)
│   │   ├── 01000165_V01epi.txt    # epi dataset
│   │   └── 01000165_V01t1.txt     # t1 dataset
│   └── 0100xxxx/          # Other datasets...
├── data/                  # Example data directory (single dataset)
├── outputs/               # Experiment output directory (auto-created)
│   ├── 20240823_143052/   # Experiment directory (timestamp format)
│   │   ├── checkpoints/   # Checkpoint files
│   │   ├── best_model/    # Best model
│   │   ├── training_curves/ # Training curves
│   │   └── inference_results/ # Inference results
│   └── test1/             # Custom experiment name
├── main_data.py           # Unified data processing entry (recommended)
├── train.py               # Model training script (refactored clean version)
├── model.py               # U-Net model definition
├── dataset.py             # Training dataset processing
├── loss.py                # Loss function module
├── utils/                 # Utility modules (new)
│   ├── __init__.py
│   ├── logger.py          # Logging tool (Tee class)
│   ├── experiment.py      # Experiment management
│   ├── config.py          # Configuration settings
│   ├── training.py        # Training utility functions
│   └── inference.py       # Inference utility functions
├── data.list              # Original data list file (520 lines)
├── train.list             # Training set data list (420 lines)
├── val.list               # Validation set data list (50 lines)
├── test.list              # Test set data list (50 lines, reserved)
├── legacy/                # Old tools directory (deprecated)
│   ├── batch_process.py   # Batch processing script (integrated)
│   ├── debug.py          # Debug image generation (integrated)
│   ├── label.py          # Annotation tool (integrated)
│   ├── process_3d_image.py # 3D to slice conversion (integrated)
│   ├── test_process.py   # Test script (integrated)
│   └── regenerate_*.py   # Debug tools (deprecated)
└── CLAUDE.md             # This project documentation
```

## Project divided into two parts

### 1. Offline Data Processing Part

#### main_data.py - Unified Data Processing Tool
**Function**: Complete 3D medical image data processing pipeline, integrates all old tool functions

**Usage**:
```bash
# Basic usage
python3 main_data.py raw_data output_data

# Example
python3 main_data.py raw_data output_data

# Parallel processing (recommended)
python3 main_data.py raw_data output_data --parallel --workers 4

# Custom worker threads
python3 main_data.py raw_data output_data --parallel --workers 8
```

**Input**:
- `raw_data/`: Root directory containing all subject directories
- Each subject directory contains:
  - `*_V01csf.txt`: CSF annotation file (4-column format: x y z value)
  - `*_V01epi.txt`: EPI data file (4-column format: x y z value)
  - `*_V01t1.txt`: T1 data file (4-column format: x y z value)

**Data Specifications**:
- **Spatial Dimensions**: 64×64×64 voxels
- **Value Range**: Raw data automatically normalized to 0-1
- **File Format**: 4-column format `x y z value`

### 2. Model Training Part

#### train.py - U-Net Model Training
**Function**: Train U-Net model for point detection based on processed data, using training and validation sets

**Data Split**:
- **train.list**: 420 samples (training set)
- **val.list**: 50 samples (validation set)
- **test.list**: 50 samples (test set, reserved)

**Usage**:
```bash
# Basic training (auto-generates timestamp experiment name)
python3 train.py --train_list train.list --val_list val.list --epochs 100

# Specify experiment name
python3 train.py --train_list train.list --val_list val.list --exp_name my_experiment

# Custom loss function and parameters
python3 train.py --train_list train.list --val_list val.list --loss focal --focal_alpha 0.5 --focal_gamma 1.5

# Resume training from checkpoint (use same experiment name)
python3 train.py --train_list train.list --val_list val.list --exp_name my_experiment

# Resume from specific checkpoint
python3 train.py --train_list train.list --val_list val.list --resume outputs/my_experiment/checkpoints/latest_checkpoint.pth
```

**Training Features**:
- **Experiment Management**: All outputs organized by experiment name in `outputs/` directory
- **Resume Training**: Support automatic recovery based on experiment name
- **Data Augmentation**: Random translation ±3 pixels, random scaling ±10%
- **Early Stopping**: Auto-stop when validation loss doesn't improve for 10 epochs
- **Multiple Loss Functions**: Support GCE, BCE, MAE, MSE, Focal Loss
- **Memory Optimization**: Small datasets loaded directly to device memory, zero-copy transfer
- **Validation Results**: Auto-save inference results every 10 epochs
- **Checkpoint Management**: Save checkpoint every 5 epochs, reduce storage usage

**CLI Parameters**:
- `--exp_name`: Experiment name (default: current timestamp format)
- `--loss`: Loss function type (gce/bce/mae/mse/focal)
- `--gce_q`: GCE loss q parameter (default 0.4)
- `--focal_alpha`, `--focal_gamma`: Focal Loss parameters
- `--resume`: Specify checkpoint path to resume training

**Output File Structure**:
```
outputs/{exp_name}/
├── best_model/best_model.pth      # Best model weights
├── best_model/best_epoch_{N}.pth  # Symbolic link pointing to best epoch checkpoint
├── checkpoints/
│   ├── latest_checkpoint.pth      # Latest checkpoint
│   └── checkpoint_epoch_{N}.pth   # Checkpoints every 5 epochs
├── training_curves/training_curves.png  # Training loss curves
└── inference_results/             # Inference results
    ├── final/                     # Final validation results (using best model)
    ├── train/
    └── val/
```

### 3. Model Evaluation Part

#### eval.py - Best Model Evaluation Tool
**Function**: Complete evaluation of validation set using trained best model, calculate IoU metrics

**Usage**:
```bash
# Basic usage (4-point discretization)
python3 eval.py gce

# Use 3-point discretization
python3 eval.py gce --point_num 3
```

**Evaluation Features**:
- **Auto Loading**: Auto-load best model weights for specified experiment
- **Complete Inference**: Full inference on all validation samples
- **Result Saving**: Save both raw inference results and discretized results
- **IoU Calculation**: Calculate IoU (Intersection/Union) between predicted and ground truth points
- **Detailed Statistics**: Show IoU for each sample and average IoU percentage

**CLI Parameters**:
- `exp_name`: Experiment name (e.g. gce)
- `--point_num`: Top-k points for discretization (3 or 4, default 4)
- `--batch_eval`: Enable batch evaluation mode, evaluate all checkpoints
- `--data_list`: Specify data list file (default val.list)
- `--loss_type`: Loss function type (for loss calculation, default focal)
- `--gce_q`: GCE loss q parameter (default 0.4)
- `--sigma`: Gaussian smoothing sigma parameter (default 2.0)
- `--num_workers`: Number of processes for batch evaluation (default 8)

**Output File Structure**:
```
outputs/{exp_name}/inference_results/val/best_{k}/
├── {sample_id}.txt          # Raw inference results (heatmap values)
├── {sample_id}_post.txt     # Discretized results (top-k is 1, others are 0)
```

**Evaluation Output Example**:
```
Sample 01000001: 4 predicted points, 3 GT points, IoU: 0.750
Sample 01000002: 4 predicted points, 2 GT points, IoU: 0.500
...
=== Evaluation Results ===
Experiment: gce
Discretization points: 4
Validation samples: 50
Average IoU: 0.6543 (65.43%)
Results saved to: outputs/gce/inference_results/val/best_4/
```

## Dependencies
```bash
pip install numpy matplotlib Pillow scipy torch torchvision scikit-learn tqdm
```

## Workflow

### 1. Data Preparation
Ensure correct data directory structure:
```
raw_data/
├── 01000001/
│   ├── 01000001_V01csf.txt
│   ├── 01000001_V01epi.txt
│   └── 01000001_V01t1.txt
└── ...
```

### 2. Generate Training Data Lists
After processing data with `main_data.py`, system automatically creates:
- `train.list` - Training set (420 samples)
- `val.list` - Validation set (50 samples)
- `test.list` - Test set (50 samples, reserved)

### 3. Start Training
```bash
python3 train.py --train_list train.list --val_list val.list --epochs 100
```

### 4. Check Results
- Model weights: Check `best_model/best_model.pth`
- Training process: View `training_curves/training_curves.png`
- Validation inference: Check inference results in `inference_results/`


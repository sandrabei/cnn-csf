"""Configuration and initialization tools"""
import torch
import torch.nn as nn
from model import LightweightUNet
from loss import get_loss_function
from dataset import MedicalImageDataset, RandomTransform
from torch.utils.data import DataLoader


def setup_device(device_str=None):
    """Set up computing device"""
    if device_str:
        if device_str.lower() == 'cpu':
            device = torch.device('cpu')
        elif device_str.lower() == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print(f"Warning: Requested device '{device_str}' not available, using auto-detect")
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return device


def setup_loss_function(loss_type, **kwargs):
    """Set up loss function"""
    print("\n=== Loss Function Configuration ===")
    try:
        if loss_type.lower() == 'gce':
            criterion = get_loss_function(loss_type=loss_type, gce_q=kwargs.get('gce_q', 0.4))
            print(f"Loss Function: Generalized Cross-Entropy (GCE)")
            print(f"  q parameter: {kwargs.get('gce_q', 0.4)}")
            print(f"  Description: Robust loss for noisy labels, q∈(0,1]")
        elif loss_type.lower() == 'bce':
            criterion = get_loss_function(loss_type=loss_type)
            print(f"Loss Function: Binary Cross-Entropy (BCE)")
            print(f"  Description: Standard cross-entropy loss")
        elif loss_type.lower() == 'mae':
            criterion = get_loss_function(loss_type=loss_type)
            print(f"Loss Function: Mean Absolute Error (MAE/L1 Loss)")
            print(f"  Description: L1 loss, robust to outliers")
        elif loss_type.lower() == 'mse':
            criterion = get_loss_function(loss_type=loss_type)
            print(f"Loss Function: Mean Squared Error (MSE)")
            print(f"  Description: L2 loss, sensitive to large errors")
        elif loss_type.lower() == 'focal':
            criterion = get_loss_function(loss_type=loss_type, 
                                        focal_alpha=kwargs.get('focal_alpha', 0.25), 
                                        focal_gamma=kwargs.get('focal_gamma', 2.0))
            print(f"Loss Function: Focal Loss")
            print(f"  alpha: {kwargs.get('focal_alpha', 0.25)} (positive class weight)")
            print(f"  gamma: {kwargs.get('focal_gamma', 2.0)} (focusing parameter)")
            print(f"  Description: Focuses on hard examples, handles class imbalance")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    except ValueError as e:
        print(f"Warning: {e}, using default GCE")
        criterion = get_loss_function(loss_type='gce', gce_q=0.4)
        print(f"Loss Function: Generalized Cross-Entropy (GCE) [Default]")
        print(f"  q parameter: 0.4")
    print("=" * 35)
    return criterion


def setup_model_and_optimizer(device, learning_rate=1e-3):
    """Set up model and optimizer"""
    model = LightweightUNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_dtype = next(model.parameters()).dtype
    
    # Calculate estimated file size
    bytes_per_param = 4 if param_dtype == torch.float32 else 2 if param_dtype == torch.float16 else 8
    estimated_file_size_mb = (total_params * bytes_per_param) / (1024 * 1024)
    
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter data type: {param_dtype}")
    print(f"Estimated model file size: {estimated_file_size_mb:.2f} MB")
    print(f"Memory per parameter: {bytes_per_param} bytes")
    print("=" * 30)
    
    return model, optimizer


def setup_datasets(train_list_path, val_list_path, device, batch_size=32, sigma=2.0):
    """Set up dataset and data loader"""
    transform = RandomTransform(shift_range=3, scale_range=0.1)
    train_dataset = MedicalImageDataset(train_list_path, sigma=sigma, transform=transform, device=str(device), cache_dir='data_cache')
    val_dataset = MedicalImageDataset(val_list_path, sigma=sigma, transform=None, device=str(device), cache_dir='data_cache')

    # Since data is already directly on device, use single-threaded DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    # Print dataset sizes for debugging
    print(f"\n=== Dataset Info ===")
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Training batches: {len(train_loader)} batches")
    print(f"Validation batches: {len(val_loader)} batches")
    print("=" * 30)
    
    return train_loader, val_loader, train_dataset, val_dataset


def setup_datasets_finetune(train_list_path, device, batch_size=20, sigma=2.0):
    """
    Set up dataset for finetune (training set only)
    """
    transform = RandomTransform(shift_range=3, scale_range=0.1)
    train_dataset = MedicalImageDataset(train_list_path, sigma=sigma, transform=transform, device=str(device), cache_dir='data_cache')

    # Since data is already directly on device, use single-threaded DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)

    # Print dataset sizes for debugging
    print(f"\n=== Dataset Info ===")
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Training batches: {len(train_loader)} batches")
    print("=" * 30)

    return train_loader, train_dataset
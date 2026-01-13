"""Configuration and initialization tools."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model import LightweightUNet
from ..loss import get_loss_function
from ..data import MedicalImageDataset, RandomTransform


def setup_device(device_str=None):
    """
    Set up computing device.

    Args:
        device_str: Device string ('cpu', 'cuda', 'mps'), or None for auto-detect

    Returns:
        torch.device: The configured device
    """
    if device_str:
        if device_str.lower() == 'cpu':
            device = torch.device('cpu')
        elif device_str.lower() == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_str.lower() == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
    return device


def setup_loss_function(loss_type='gce', **kwargs):
    """
    Set up loss function.

    Args:
        loss_type: Type of loss function ('gce', 'bce', 'mae', 'mse', 'focal')
        **kwargs: Additional parameters for the loss function

    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()

    if loss_type == 'gce':
        q = kwargs.get('gce_q', 0.4)
        return get_loss_function(loss_type=loss_type, gce_q=q)
    elif loss_type == 'bce':
        return get_loss_function(loss_type=loss_type)
    elif loss_type == 'mae':
        return get_loss_function(loss_type=loss_type)
    elif loss_type == 'mse':
        return get_loss_function(loss_type=loss_type)
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return get_loss_function(loss_type=loss_type, focal_alpha=alpha, focal_gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def setup_model_and_optimizer(device, learning_rate=1e-3):
    """
    Set up model and optimizer.

    Args:
        device: Target device
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (model, optimizer)
    """
    model = LightweightUNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def setup_datasets(train_list_path, val_list_path, device, batch_size=32, sigma=2.0):
    """
    Set up datasets and data loaders.

    Args:
        train_list_path: Path to training data list
        val_list_path: Path to validation data list
        device: Target device
        batch_size: Batch size
        sigma: Gaussian sigma for heatmap generation

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    transform = RandomTransform(shift_range=3, scale_range=0.1)
    train_dataset = MedicalImageDataset(train_list_path, sigma=sigma, transform=transform,
                                       device=str(device), cache_dir='data_cache')
    val_dataset = MedicalImageDataset(val_list_path, sigma=sigma, transform=None,
                                     device=str(device), cache_dir='data_cache')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    return train_loader, val_loader, train_dataset, val_dataset


def setup_datasets_finetune(train_list_path, device, batch_size=20, sigma=2.0):
    """
    Set up dataset for finetune (training set only).

    Args:
        train_list_path: Path to training data list
        device: Target device
        batch_size: Batch size
        sigma: Gaussian sigma for heatmap generation

    Returns:
        Tuple of (train_loader, train_dataset)
    """
    transform = RandomTransform(shift_range=3, scale_range=0.1)
    train_dataset = MedicalImageDataset(train_list_path, sigma=sigma, transform=transform,
                                       device=str(device), cache_dir='data_cache')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)

    return train_loader, train_dataset

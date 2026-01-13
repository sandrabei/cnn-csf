"""Training utilities for CNN-CSF library."""
import torch
import torch.nn as nn
from tqdm import tqdm
import time


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, epoch_time)
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    start_time = time.time()

    with tqdm(dataloader, desc="Training", leave=False) as batch_pbar:
        for batch_idx, (inputs, targets) in enumerate(batch_pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            batch_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})

    epoch_time = time.time() - start_time
    return total_loss / num_batches, epoch_time


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate one epoch.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, epoch_time)
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    start_time = time.time()

    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", leave=False) as batch_pbar:
            for inputs, targets in batch_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_loss = loss.item()
                total_loss += batch_loss

                batch_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})

    epoch_time = time.time() - start_time
    return total_loss / num_batches, epoch_time

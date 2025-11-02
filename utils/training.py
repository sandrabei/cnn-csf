"""Training utility functions"""
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt


def train_epoch(model, dataloader, criterion, optimizer, device, collect_samples=False):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    total_samples = 0
    
    collected_inputs = []
    collected_targets = []
    collected_predictions = []
    collected_count = 0
    max_collect = 10 if collect_samples else 0
    
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
            total_samples += inputs.size(0)
            
            # Collect samples for dumping if needed
            if collect_samples and collected_count < max_collect:
                remaining_needed = max_collect - collected_count
                take_samples = min(remaining_needed, inputs.size(0))
                
                collected_inputs.append(inputs[:take_samples].cpu())
                collected_targets.append(targets[:take_samples].cpu())
                collected_predictions.append(outputs[:take_samples].cpu())
                collected_count += take_samples
            
            batch_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
    
    epoch_time = time.time() - start_time
    
    if collect_samples and collected_count > 0:
        all_inputs = torch.cat(collected_inputs, dim=0) if collected_inputs else torch.empty(0)
        all_targets = torch.cat(collected_targets, dim=0) if collected_targets else torch.empty(0)
        all_predictions = torch.cat(collected_predictions, dim=0) if collected_predictions else torch.empty(0)
        return total_loss / num_batches, all_predictions, all_targets, total_samples, epoch_time
    
    return total_loss / num_batches, None, None, total_samples, epoch_time


def validate_epoch(model, dataloader, criterion, device, collect_samples=False):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    total_samples = 0
    
    collected_inputs = []
    collected_targets = []
    collected_predictions = []
    collected_count = 0
    max_collect = 10 if collect_samples else 0
    
    start_time = time.time()
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", leave=False) as batch_pbar:
            for inputs, targets in batch_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_loss = loss.item()
                total_loss += batch_loss
                total_samples += inputs.size(0)
                
                # Collect samples for dumping if needed
                if collect_samples and collected_count < max_collect:
                    remaining_needed = max_collect - collected_count
                    take_samples = min(remaining_needed, inputs.size(0))
                    
                    collected_inputs.append(inputs[:take_samples].cpu())
                    collected_targets.append(targets[:take_samples].cpu())
                    collected_predictions.append(outputs[:take_samples].cpu())
                    collected_count += take_samples
                
                batch_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
    
    epoch_time = time.time() - start_time
    
    if collect_samples and collected_count > 0:
        all_inputs = torch.cat(collected_inputs, dim=0) if collected_inputs else torch.empty(0)
        all_targets = torch.cat(collected_targets, dim=0) if collected_targets else torch.empty(0)
        all_predictions = torch.cat(collected_predictions, dim=0) if collected_predictions else torch.empty(0)
        return total_loss / num_batches, all_predictions, all_targets, total_samples, epoch_time
    
    return total_loss / num_batches, None, None, total_samples, epoch_time


def save_training_curves(train_losses, val_losses, curves_dir, epoch):
    """Save training curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Training Curves - Epoch {epoch + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(curves_dir / 'training_curves.png')
    plt.close()


def save_checkpoint(checkpoint, checkpoint_dir, epoch, is_latest=True):
    """Save checkpoint"""
    saved = False

    if is_latest:
        latest_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

    if (epoch + 1) % 5 == 0:
        epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, epoch_path)
        saved = True

    return saved
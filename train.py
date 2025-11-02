#!/usr/bin/env python3
"""
U-Net model training script - Refactored version
"""

import argparse
import time
import torch
from pathlib import Path
from tqdm import tqdm

from utils.logger import Tee
from utils.experiment import setup_experiment_dir, print_experiment_info
from utils.config import setup_device, setup_loss_function, setup_model_and_optimizer, setup_datasets, setup_datasets_finetune
from utils.training import train_epoch, validate_epoch, save_training_curves, save_checkpoint
from utils.inference import run_final_inference


def train_model(train_list_path, val_list_path=None, num_epochs=100, batch_size=32, learning_rate=1e-3,
                resume_path=None, device_str=None, num_workers=None, loss_type='gce', gce_q=0.4,
                focal_alpha=0.25, focal_gamma=2.0, sigma=2.0, exp_name=None, finetune_mode=False,
                finetune_base_dir='outputs_finetune'):
    """
    Main function for training U-Net model

    Args:
        train_list_path: Training data list path
        val_list_path: Validation data list path (can be None in finetune mode)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        resume_path: Path to resume training checkpoint
        device_str: Device type ('cpu' or 'mps')
        num_workers: Number of data loading worker threads (optimized to 0)
        loss_type: Loss function type
        gce_q: GCE loss q parameter
        focal_alpha: Focal Loss alpha parameter
        focal_gamma: Focal Loss gamma parameter
        sigma: Gaussian smoothing sigma parameter (default 2.0)
        exp_name: Experiment name
        finetune_mode: Whether it's finetune mode (no validation set, save checkpoint every 5 epochs)
        finetune_base_dir: Output directory for finetune mode

    Returns:
        dict: Training result information
    """
    
    # Set up experiment directory
    if finetune_mode:
        # Finetune mode: use custom directory structure
        from pathlib import Path
        base_output_dir = Path(finetune_base_dir)
        base_output_dir.mkdir(exist_ok=True)

        if exp_name is None:
            exp_name = time.strftime('%Y%m%d_%H%M%S')

        exp_dir = base_output_dir / exp_name
        sub_dirs = {
            'exp_dir': exp_dir,
            'checkpoint_dir': exp_dir / 'checkpoints',
            'best_model_dir': exp_dir / 'best_model',
            'curves_dir': exp_dir / 'training_curves',
            'inference_dir': exp_dir / 'inference_results',
            'log_file': exp_dir / 'training.log'
        }

        # Create all directories (except log_file which is a file)
        for key, dir_path in sub_dirs.items():
            if key != 'log_file' and isinstance(dir_path, Path):
                dir_path.mkdir(parents=True, exist_ok=True)
    else:
        # Normal training mode
        exp_name, exp_dir, sub_dirs = setup_experiment_dir(exp_name)

    print_experiment_info(exp_name, exp_dir, sub_dirs)
    
    # Set up logging
    tee = Tee(str(sub_dirs['log_file']))
    
    # Set up device
    device = setup_device(device_str)

    # Set up loss function
    loss_kwargs = {'gce_q': gce_q, 'focal_alpha': focal_alpha, 'focal_gamma': focal_gamma}
    criterion = setup_loss_function(loss_type, **loss_kwargs)

    # Set up model and optimizer
    model, optimizer = setup_model_and_optimizer(device, learning_rate)

    # Set up datasets
    if finetune_mode:
        # Finetune mode: only training set
        train_loader, train_dataset = setup_datasets_finetune(
            train_list_path, device, batch_size, sigma)
        val_loader = None
        val_dataset = None
    else:
        # Normal training mode: has training and validation sets
        train_loader, val_loader, train_dataset, val_dataset = setup_datasets(
            train_list_path, val_list_path, device, batch_size, sigma)
    
    # Print all training parameters
    mode_name = "Finetune" if finetune_mode else "Training"
    print(f"\n=== {mode_name} Configuration ===")
    print(f"epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"loss_type: {loss_type}")
    if not finetune_mode:
        print(f"gce_q: {gce_q}")
    print(f"focal_alpha: {focal_alpha}")
    print(f"focal_gamma: {focal_gamma}")
    print(f"sigma: {sigma}")
    print(f"device: {device}")
    print(f"train_samples: {len(train_dataset)}")
    if not finetune_mode and val_dataset:
        print(f"val_samples: {len(val_dataset)}")
    print("=" * 35)
    
    # Initialize training state
    if finetune_mode:
        # Finetune mode: no validation set, no early stopping needed
        train_losses = []
        start_epoch = 0
    else:
        # Normal training mode: has validation set and early stopping
        best_val_loss = float('inf')
        best_epoch = 0  # Save best epoch
        patience = 10
        patience_counter = 0
        train_losses = []
        val_losses = []
        start_epoch = 0
    
    # Checkpoint recovery logic
    checkpoint_to_load = None
    if resume_path:
        if Path(resume_path).exists():
            checkpoint_to_load = resume_path
            print(f"Using explicit resume path: {resume_path}")
        else:
            print(f"Warning: Specified resume path '{resume_path}' not found")
    else:
        # Try to recover from experiment directory (normal training mode only)
        if not finetune_mode:
            latest_checkpoint = sub_dirs['checkpoint_dir'] / 'latest_checkpoint.pth'
            if latest_checkpoint.exists():
                checkpoint_to_load = str(latest_checkpoint)
                print(f"Found checkpoint in experiment directory: {latest_checkpoint}")
            else:
                print(f"No checkpoint found in experiment directory '{exp_name}', starting fresh")
    
    if checkpoint_to_load and Path(checkpoint_to_load).exists():
        if finetune_mode:
            print(f"Loading pretrained model for finetune: {checkpoint_to_load}")
            checkpoint = torch.load(checkpoint_to_load, map_location=device)

            # Finetune mode: only load model weights, don't load training state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model state dict from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights directly")

            print(f"Starting finetune from scratch with pretrained weights")
        else:
            print(f"Resuming training from checkpoint: {checkpoint_to_load}")
            checkpoint = torch.load(checkpoint_to_load, map_location=device)

            # Verify experiment match
            if 'exp_name' in checkpoint and checkpoint['exp_name'] != exp_name:
                print(f"Warning: Checkpoint experiment '{checkpoint['exp_name']}' doesn't match current exp_name '{exp_name}'")
                print("Starting training from scratch")
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                best_epoch = checkpoint.get('best_epoch', 0)
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                patience_counter = checkpoint.get('patience_counter', 0)

                print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}, best epoch: {best_epoch}")
    else:
        mode_name = "finetune" if finetune_mode else "training"
        print(f"Starting {mode_name} from scratch for experiment '{exp_name}'")
    
    # Main training loop
    loop_desc = "Finetuning" if finetune_mode else "Training"
    with tqdm(range(start_epoch, num_epochs), desc=loop_desc, initial=start_epoch) as pbar:
        for epoch in pbar:
            # Training
            train_start = time.time()
            train_loss, _, _, train_samples, train_epoch_time = train_epoch(
                model, train_loader, criterion, optimizer, device)
            train_total_time = time.time() - train_start

            train_losses.append(train_loss)

            if not finetune_mode:
                # Normal training mode: perform validation
                val_start = time.time()
                val_loss, _, _, val_samples, val_epoch_time = validate_epoch(
                    model, val_loader, criterion, device)
                val_total_time = time.time() - val_start
                val_losses.append(val_loss)
            else:
                # Finetune mode: no validation
                val_loss = None
                val_samples = 0
                val_total_time = 0
            
            # Calculate training speed
            train_speed = train_samples / train_total_time if train_total_time > 0 else 0
            val_speed = val_samples / val_total_time if val_total_time > 0 else 0
            
            # Print detailed log
            print(f"Epoch {epoch + 1}/{num_epochs} completed:")
            print(f"  Training:   {train_samples} images in {train_total_time:.2f}s ({train_speed:.2f} images/sec)")
            if not finetune_mode:
                print(f"  Validation: {val_samples} images in {val_total_time:.2f}s ({val_speed:.2f} images/sec)")
                print(f"  Loss: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}")
                pbar.set_postfix({
                    'train': f'{train_loss:.4f}',
                    'val': f'{val_loss:.4f}',
                    'best': f'{best_val_loss:.4f}'
                })
            else:
                print(f"  Loss: {train_loss:.4f}")
                pbar.set_postfix({'train': f'{train_loss:.4f}'})
            print()
            
            # Save training curves
            if finetune_mode:
                save_training_curves(train_losses, [], sub_dirs['curves_dir'], epoch)
            else:
                save_training_curves(train_losses, val_losses, sub_dirs['curves_dir'], epoch)

            # Save inference results every 10 epochs (normal training mode only)
            if not finetune_mode and (epoch + 1) % 10 == 0:
                # Validation set inference
                _, val_preds, val_targets, _, _ = validate_epoch(
                    model, val_loader, criterion, device, collect_samples=True)
                if val_preds is not None and val_targets is not None:
                    val_sample_ids = [Path(val_dataset.data_list.iloc[idx]['output']).parent.name 
                                    for idx in range(val_preds.size(0))]
                    from utils.inference import dump_inference_results
                    dump_inference_results(val_preds, val_targets, str(sub_dirs['inference_dir']), 
                                         epoch + 1, "val", val_sample_ids)
                
                # Training set inference
                _, train_preds, train_targets, _, _ = train_epoch(
                    model, train_loader, criterion, optimizer, device, collect_samples=True)
                if train_preds is not None and train_targets is not None:
                    train_sample_ids = [Path(train_dataset.data_list.iloc[idx]['output']).parent.name 
                                      for idx in range(train_preds.size(0))]
                    dump_inference_results(train_preds, train_targets, str(sub_dirs['inference_dir']), 
                                         epoch + 1, "train", train_sample_ids)
            
            if not finetune_mode:
                # Normal training mode: early stopping and best model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1  # Record best epoch (counting from 1)
                    best_model_path = sub_dirs['best_model_dir'] / 'best_model.pth'
                    torch.save(model.state_dict(), best_model_path)

                    # Create symbolic link pointing to corresponding epoch checkpoint
                    epoch_checkpoint = sub_dirs['checkpoint_dir'] / f'checkpoint_epoch_{epoch + 1}.pth'
                    if epoch_checkpoint.exists():
                        symlink_path = sub_dirs['best_model_dir'] / f'best_epoch_{epoch + 1}.pth'
                        # Delete old symbolic links (if exist)
                        for old_link in sub_dirs['best_model_dir'].glob('best_epoch_*.pth'):
                            old_link.unlink(missing_ok=True)
                        # Create new symbolic link
                        symlink_path.symlink_to(epoch_checkpoint.name)

                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.set_description(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save checkpoint
            if finetune_mode:
                # Finetune mode: save checkpoint every 5 epochs
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'exp_name': exp_name,
                    'loss_type': loss_type
                }

                # save_checkpoint function will automatically determine if saving is needed
                saved = save_checkpoint(checkpoint, sub_dirs['checkpoint_dir'], epoch)
                if saved:
                    print(f"Saved checkpoint for epoch {epoch + 1}")
            else:
                # Normal training mode: save checkpoint according to original logic
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'patience_counter': patience_counter,
                    'exp_name': exp_name
                }

                save_checkpoint(checkpoint, sub_dirs['checkpoint_dir'], epoch)
    
    # Final inference (normal training mode only)
    final_samples = 0
    if not finetune_mode and val_dataset:
        # Load best model weights
        best_model = model
        best_model_path = sub_dirs['best_model_dir'] / 'best_model.pth'
        if best_model_path.exists():
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))

        final_samples = run_final_inference(best_model, val_dataset, device, sub_dirs['inference_dir'], best_epoch)

    # Cleanup
    tee.close()

    # Print summary
    if finetune_mode:
        print(f"\n=== Finetune Summary ===")
        print(f"Experiment: {exp_dir}")
        print(f"Total epochs: {len(train_losses)}")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        return {
            'train_losses': train_losses,
            'exp_dir': str(exp_dir),
            'total_epochs': len(train_losses)
        }
    else:
        print(f"\n=== Training Summary ===")
        print(f"Experiment: {exp_dir}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Total epochs: {len(train_losses)}")
        print(f"Final validation inference: {final_samples} samples (using best model from epoch {best_epoch})")
        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'exp_dir': str(exp_dir),
            'final_samples': final_samples
        }


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for medical image point detection')
    parser.add_argument('--train_list', default='train.list', help='Path to train.list file')
    parser.add_argument('--val_list', default='val.list', help='Path to val.list file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from (optional)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps'], 
                       help='Device to use for training (cpu or mps)')
    parser.add_argument('--loss', type=str, default='gce', 
                       choices=['gce', 'bce', 'mae', 'mse', 'focal'], 
                       help='Loss function type')
    parser.add_argument('--gce_q', type=float, default=0.4, help='q parameter for GCE loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha parameter for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma parameter for Focal Loss')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian smoothing sigma parameter (default: 2.0)')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (default: YYYYMMDD_HHMMSS)')
    parser.add_argument('--finetune', action='store_true', help='Enable finetune mode (no validation, save checkpoint every 5 epochs)')
    parser.add_argument('--finetune_base_dir', type=str, default='outputs_finetune', help='Base directory for finetune experiments')

    args = parser.parse_args()

    # Finetune mode configuration
    if args.finetune:
        if args.val_list == 'val.list':
            args.val_list = None  # Finetune mode doesn't need validation set
        if args.exp_name is None:
            args.exp_name = 'v15_focal'  # Default experiment name
        if args.resume is None:
            args.resume = 'outputs/v15_focal/best_model/best_model.pth'  # Default pretrained model
        # Remove restriction to force focal loss, allow users to specify loss type

    result = train_model(
        args.train_list,
        args.val_list,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_path=args.resume,
        device_str=args.device,
        loss_type=args.loss,
        gce_q=args.gce_q,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sigma=args.sigma,
        exp_name=args.exp_name,
        finetune_mode=args.finetune,
        finetune_base_dir=args.finetune_base_dir
    )


if __name__ == '__main__':
    main()

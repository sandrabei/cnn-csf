"""Fine-tuning API for CNN-CSF library."""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .model import LightweightUNet
from .loss import get_loss_function
from .data import MedicalImageDataset
from .utils import setup_device, train_epoch, validate_epoch
from torch.utils.data import DataLoader


def finetune(
    train_list_path,
    checkpoint_path,
    output_dir,
    val_list_path=None,
    epochs=50,
    batch_size=20,
    learning_rate=1e-4,
    loss_type='focal',
    sigma=2.0,
    focal_alpha=0.5,
    focal_gamma=1.5,
    device='auto',
    save_every=5,
    early_stopping_patience=10,
    verbose=True
):
    """
    Fine-tune a pre-trained model on custom data.

    This function fine-tunes an existing model on a new dataset. It supports
    both training-only mode (no validation) and training+validation mode.

    Args:
        train_list_path: Path to training data list file (CSV format: input1,input2,output)
        checkpoint_path: Path to pre-trained model checkpoint (.pth)
        output_dir: Directory to save fine-tuned models and logs
        val_list_path: Path to validation data list (optional, if None, no validation)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        loss_type: Loss function type ('gce', 'bce', 'mae', 'mse', 'focal')
        sigma: Gaussian sigma for heatmap generation
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
        save_every: Save checkpoint every N epochs
        early_stopping_patience: Early stopping patience (None to disable)
        verbose: Print progress information

    Returns:
        dict: Training history containing:
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch (if validation)
            - best_val_loss: Best validation loss achieved (if validation)
            - best_epoch: Epoch number with best validation loss

    Example:
        >>> from cnn_csf import finetune
        >>>
        >>> # Fine-tune without validation
        >>> history = finetune(
        ...     train_list_path='my_train.list',
        ...     checkpoint_path='pretrained_model.pth',
        ...     output_dir='finetuned_model',
        ...     epochs=30
        ... )
        >>>
        >>> # Fine-tune with validation
        >>> history = finetune(
        ...     train_list_path='my_train.list',
        ...     val_list_path='my_val.list',
        ...     checkpoint_path='pretrained_model.pth',
        ...     output_dir='finetuned_model',
        ...     epochs=50,
        ...     loss_type='focal'
        ... )

    Note:
        - Data list files should be CSV format: input1_path,input2_path,output_path
        - The fine-tuned model will be saved to {output_dir}/best_model/best_model.pth
        - If no validation set is provided, the last epoch model is saved as best
    """
    # Setup device
    device = setup_device(device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    model = LightweightUNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Setup loss function
    criterion = get_loss_function(
        loss_type=loss_type,
        gce_q=0.4,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup datasets
    if verbose:
        print(f"Loading training data from: {train_list_path}")

    train_dataset = MedicalImageDataset(
        train_list_path,
        sigma=sigma,
        transform=None,
        device=str(device),
        cache_dir=str(output_dir / 'cache')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    if val_list_path is not None:
        val_dataset = MedicalImageDataset(
            val_list_path,
            sigma=sigma,
            transform=None,
            device=str(device),
            cache_dir=str(output_dir / 'cache')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    else:
        val_loader = None

    if verbose:
        print(f"Training samples: {len(train_dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Device: {device}")
        print(f"Loss function: {loss_type}")

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    has_validation = val_loader is not None

    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        if verbose:
            print(f"Train Loss: {train_loss:.6f}")

        # Validation
        if has_validation:
            val_loss, _ = validate_epoch(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            if verbose:
                print(f"Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                best_model_dir = output_dir / 'best_model'
                best_model_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_model_dir / 'best_model.pth')

                if verbose:
                    print(f"Saved best model (val_loss: {val_loss:.6f})")

            # Early stopping
            if early_stopping_patience and (epoch + 1 - best_epoch) >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            # No validation, save last model as best
            best_model_dir = output_dir / 'best_model'
            best_model_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, best_model_dir / 'best_model.pth')

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = output_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_losses[-1] if has_validation else None,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth')

            if verbose:
                print(f"Saved checkpoint at epoch {epoch + 1}")

    # Return training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses if has_validation else None,
        'best_val_loss': best_val_loss if has_validation else None,
        'best_epoch': best_epoch if has_validation else epochs,
    }

    if verbose:
        print(f"\nTraining completed!")
        if has_validation:
            print(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"Model saved to: {output_dir / 'best_model' / 'best_model.pth'}")

    return history

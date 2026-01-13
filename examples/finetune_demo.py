"""
Fine-tuning Demo for CNN-CSF Library

This script demonstrates how to use the finetune() function to
fine-tune a pre-trained model on custom data.

It will use real data if available in test_data/, otherwise provide
a template for users to adapt.
"""
from pathlib import Path
from cnn_csf import finetune


def demo_with_real_data():
    """Run finetune demo with real data (for testing)."""
    print("=" * 60)
    print("CNN-CSF Library - Fine-tuning Demo (Real Data)")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / "test_data" / "best_model.pth"
    output_dir = project_root / "outputs" / "demo_finetune"

    # Create a small training list for demo
    train_list_path = project_root / "demo_train.list"

    # Check if test data exists
    test_sample = project_root / "test_data" / "01000165"
    if not test_sample.exists():
        raise FileNotFoundError(f"Test data not found: {test_sample}")

    print(f"\nCreating training list: {train_list_path}")
    print("(Using the same sample multiple times for demo purposes)")
    with open(train_list_path, 'w') as f:
        for i in range(5):
            f.write(f"{test_sample}/epi.txt,{test_sample}/t1.txt,{test_sample}/csf.txt\n")

    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Training samples: 5")
    print(f"  Epochs: 3 (demo mode)")
    print(f"  Output directory: {output_dir}")

    # Run fine-tuning
    print("\nStarting fine-tuning...")
    history = finetune(
        train_list_path=str(train_list_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        epochs=3,  # Just 3 epochs for demo
        batch_size=2,
        learning_rate=1e-4,
        loss_type='focal',
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)
    print(f"Training losses: {history['train_losses']}")
    print(f"Best model saved to: {output_dir}/best_model/best_model.pth")

    # Cleanup demo file
    train_list_path.unlink()


def demo_template_without_validation():
    """
    Template: Fine-tune WITHOUT validation set.

    Modify the paths below to use your own data.
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Template (No Validation)")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # ===== MODIFY THESE PATHS =====
    checkpoint_path = "path/to/your/pretrained_model.pth"
    train_list_path = "path/to/your_train.list"
    output_dir = "outputs/your_finetuned_model"
    # ===============================

    # Check if paths are just placeholders
    if checkpoint_path.startswith("path/to/"):
        print("\nTemplate mode - please modify the paths in this script:")
        print("  1. checkpoint_path: Path to your pre-trained model")
        print("  2. train_list_path: Path to your training data list")
        print("  3. output_dir: Where to save the fine-tuned model")
        print("\nData list format (CSV):")
        print("  input1_path,input2_path,output_path")
        print("  data/epi/sample1.txt,data/t1/sample1.txt,data/label/sample1.txt")
        print("  ...")
        return

    print(f"\nRunning fine-tuning...")
    history = finetune(
        train_list_path=train_list_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        epochs=30,  # Adjust as needed
        batch_size=20,  # Adjust based on your GPU memory
        learning_rate=1e-4,  # Usually smaller than pre-training learning rate
        loss_type='focal',
        verbose=True
    )

    print("\nTraining completed!")
    print(f"Best model saved to: {output_dir}/best_model/best_model.pth")


def demo_template_with_validation():
    """
    Template: Fine-tune WITH validation set.

    Modify the paths below to use your own data.
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Template (With Validation)")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # ===== MODIFY THESE PATHS =====
    checkpoint_path = "path/to/your/pretrained_model.pth"
    train_list_path = "path/to/your_train.list"
    val_list_path = "path/to/your_val.list"
    output_dir = "outputs/your_finetuned_model_with_val"
    # ===============================

    if checkpoint_path.startswith("path/to/"):
        print("\nTemplate mode - please modify the paths in this script:")
        print("  1. checkpoint_path: Path to your pre-trained model")
        print("  2. train_list_path: Path to your training data list")
        print("  3. val_list_path: Path to your validation data list")
        print("  4. output_dir: Where to save the fine-tuned model")
        print("\nWith validation, the best model is selected based on validation loss.")
        print("Early stopping is enabled by default (patience=10 epochs).")
        return

    print(f"\nRunning fine-tuning with validation...")
    history = finetune(
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        epochs=50,
        batch_size=20,
        learning_rate=1e-4,
        loss_type='focal',
        focal_alpha=0.5,  # Focal loss parameters
        focal_gamma=1.5,
        early_stopping_patience=10,
        verbose=True
    )

    print("\nTraining completed!")
    print(f"Best val_loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")
    print(f"Best model saved to: {output_dir}/best_model/best_model.pth")


def demo_template_custom_parameters():
    """
    Template: Fine-tune with custom parameters.

    This shows various parameters you can customize.
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Template (Custom Parameters)")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # ===== MODIFY THESE PATHS =====
    checkpoint_path = "path/to/your/pretrained_model.pth"
    train_list_path = "path/to/your_train.list"
    val_list_path = "path/to/your_val.list"
    output_dir = "outputs/your_custom_finetune"
    # ===============================

    if checkpoint_path.startswith("path/to/"):
        print("\nAvailable customization options:")
        print("  epochs: Number of training epochs (default: 50)")
        print("  batch_size: Batch size (default: 20)")
        print("  learning_rate: Learning rate (default: 1e-4)")
        print("  loss_type: 'gce', 'bce', 'mae', 'mse', or 'focal' (default: 'focal')")
        print("  sigma: Gaussian sigma for heatmap generation (default: 2.0)")
        print("  focal_alpha, focal_gamma: Focal loss parameters")
        print("  early_stopping_patience: Patience for early stopping (default: 10)")
        print("  save_every: Save checkpoint every N epochs (default: 5)")
        print("  device: 'auto', 'cpu', 'cuda', or 'mps' (default: 'auto')")
        return

    print(f"\nRunning fine-tuning with custom parameters...")
    history = finetune(
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        epochs=100,                    # More epochs
        batch_size=32,                 # Larger batch size
        learning_rate=5e-5,            # Lower learning rate
        loss_type='gce',               # Use GCE loss
        sigma=2.5,                     # Larger Gaussian sigma
        early_stopping_patience=15,    # More patience
        save_every=10,                 # Save every 10 epochs
        device='auto',                 # Auto-detect GPU
        verbose=True
    )

    print("\nTraining completed!")


def demo_create_data_list():
    """
    Example: Create a data list file for fine-tuning.

    This shows how to create the .list file that finetune() expects.
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("Creating Data List File")
    print("=" * 60)

    # Suppose your data is organized like this:
    # data/
    #   sample_001/
    #     epi.txt
    #     t1.txt
    #     csf.txt
    #   sample_002/
    #     ...

    data_dir = Path("data")

    if not data_dir.exists():
        print("\nExample data structure:")
        print("  data/")
        print("    sample_001/")
        print("      epi.txt")
        print("      t1.txt")
        print("      csf.txt")
        print("    sample_002/")
        print("      ...")
        print("\nThis code would scan that directory and create train.list")
        return

    # Create list of samples
    samples = []
    for sample_dir in sorted(data_dir.glob("sample_*")):
        epi_file = sample_dir / "epi.txt"
        t1_file = sample_dir / "t1.txt"
        csf_file = sample_dir / "csf.txt"

        if all(f.exists() for f in [epi_file, t1_file, csf_file]):
            samples.append({
                'input1': str(epi_file),
                'input2': str(t1_file),
                'output': str(csf_file)
            })

    # Create DataFrame and save
    df = pd.DataFrame(samples)
    df.to_csv('train.list', header=False, index=False)

    print(f"Created data list with {len(samples)} samples")
    print(f"Saved to: train.list")

    # For validation set, you might split your data:
    # train_samples = samples[:int(len(samples) * 0.8)]
    # val_samples = samples[int(len(samples) * 0.8):]
    #
    # pd.DataFrame(train_samples).to_csv('train.list', header=False, index=False)
    # pd.DataFrame(val_samples).to_csv('val.list', header=False, index=False)


def main():
    """Main entry point - tries real data first, shows templates otherwise."""
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / "test_data" / "best_model.pth"
    test_sample = project_root / "test_data" / "01000165"

    # Check if real test data is available
    has_real_data = checkpoint_path.exists() and test_sample.exists()

    print("CNN-CSF Library - Fine-tuning Demo")
    print("=" * 60)

    if has_real_data:
        print("\nTest data detected! Running demo with real data...")
        demo_with_real_data()
    else:
        print("\nNo test data found. Showing template examples...")
        print("Modify the paths in each function to use your own data.\n")

        # Show templates
        demo_template_without_validation()
        demo_template_with_validation()
        demo_template_custom_parameters()
        demo_create_data_list()


if __name__ == "__main__":
    main()

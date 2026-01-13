"""
Inference Demo for CNN-CSF Library

This script demonstrates how to use the inference() function to run
model inference on medical image data.

It will use real data if available in test_data/, otherwise fall back
to dummy data for demonstration.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from cnn_csf import inference
from cnn_csf.utils import extract_peaks


def load_64x64_data(file_path):
    """Load 64x64 data from text file."""
    data = pd.read_csv(file_path, sep=r'\s+', header=None,
                      names=['x', 'y', 'value'])

    grid = np.zeros((64, 64))
    for _, row in data.iterrows():
        x, y, value = int(row['x']), int(row['y']), float(row['value'])
        if 0 <= x < 64 and 0 <= y < 64:
            grid[y, x] = value

    # Normalize to 0-1
    if grid.max() > 0:
        grid = grid / grid.max()

    return grid.astype(np.float32)


def demo_with_real_data():
    """Run inference demo with real medical image data."""
    print("=" * 60)
    print("CNN-CSF Library - Inference Demo (Real Data)")
    print("=" * 60)

    # Paths to test data
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "test_data" / "01000165"
    checkpoint_path = project_root / "test_data" / "best_model.pth"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test data not found: {test_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load data
    print(f"\nLoading data from: {test_dir}")
    epi_file = test_dir / "epi.txt"
    t1_file = test_dir / "t1.txt"
    csf_file = test_dir / "csf.txt"

    epi_data = load_64x64_data(epi_file)
    t1_data = load_64x64_data(t1_file)

    print(f"EPI data shape: {epi_data.shape}, range: [{epi_data.min():.4f}, {epi_data.max():.4f}]")
    print(f"T1 data shape: {t1_data.shape}, range: [{t1_data.min():.4f}, {t1_data.max():.4f}]")

    # Stack channels to create input array
    input_data = np.stack([epi_data, t1_data], axis=0)  # Shape: (2, 64, 64)
    print(f"Input shape: {input_data.shape}")

    # Run inference
    print(f"\nRunning inference with checkpoint: {checkpoint_path}")
    heatmap = inference(input_data, checkpoint_path=str(checkpoint_path))

    # Output
    print(f"\nOutput heatmap shape: {heatmap.shape}")
    print(f"Output range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Find peak locations
    heatmap_2d = heatmap.squeeze()
    flat_heatmap = heatmap_2d.flatten()
    top_4_indices = np.argpartition(flat_heatmap, -4)[-4:]
    top_4_indices = top_4_indices[np.argsort(-flat_heatmap[top_4_indices])]

    print("\n" + "=" * 60)
    print("Top 4 predicted point locations:")
    print("=" * 60)
    for i, idx in enumerate(top_4_indices, 1):
        y, x = np.unravel_index(idx, heatmap_2d.shape)
        value = heatmap_2d[y, x]
        print(f"  Point {i}: ({x:2d}, {y:2d}) with probability {value:.6f}")

    # Using extract_peaks utility
    print("\n" + "=" * 60)
    print("Using extract_peaks utility:")
    print("=" * 60)
    peaks = extract_peaks(heatmap_2d, threshold=0.1, min_distance=3)
    print(f"Found {len(peaks)} peaks above threshold:")
    for i, (x, y) in enumerate(peaks[:4], 1):
        value = heatmap_2d[y, x]
        print(f"  Peak {i}: ({x:2d}, {y:2d}) with probability {value:.6f}")

    # Compare with ground truth
    print("\n" + "=" * 60)
    print("Ground truth labels:")
    print("=" * 60)
    csf_data = pd.read_csv(csf_file, sep=r'\s+', header=None,
                          names=['x', 'y', 'value'])
    gt_points = csf_data[csf_data['value'] > 0]
    print(f"Number of GT points: {len(gt_points)}")
    for _, row in gt_points.iterrows():
        print(f"  GT point: ({int(row['x']):2d}, {int(row['y']):2d})")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demo_with_dummy_data():
    """Run inference demo with dummy data (for testing API)."""
    print("=" * 60)
    print("CNN-CSF Library - Inference Demo (Dummy Data)")
    print("=" * 60)
    print("\nNote: Using dummy data. For real results, provide test_data/ directory")

    # Create dummy input data
    print("\nCreating dummy input data...")
    epi_data = np.random.rand(64, 64)
    t1_data = np.random.rand(64, 64)

    input_data = np.stack([epi_data, t1_data], axis=0)  # Shape: (2, 64, 64)
    print(f"Input shape: {input_data.shape}")

    # For dummy data, we need a checkpoint path
    # This will fail if checkpoint doesn't exist, but demonstrates the API
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / "test_data" / "best_model.pth"

    if checkpoint_path.exists():
        print(f"\nRunning inference with checkpoint: {checkpoint_path}")
        heatmap = inference(input_data, checkpoint_path=str(checkpoint_path))

        print(f"\nOutput heatmap shape: {heatmap.shape}")
        print(f"Output range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

        # Find top peaks
        heatmap_2d = heatmap.squeeze()
        flat_heatmap = heatmap_2d.flatten()
        top_4_indices = np.argpartition(flat_heatmap, -4)[-4:]
        top_4_indices = top_4_indices[np.argsort(-flat_heatmap[top_4_indices])]

        print("\n" + "=" * 60)
        print("Top 4 predicted point locations:")
        print("=" * 60)
        for i, idx in enumerate(top_4_indices, 1):
            y, x = np.unravel_index(idx, heatmap_2d.shape)
            value = heatmap_2d[y, x]
            print(f"  Point {i}: ({x:2d}, {y:2d}) with probability {value:.6f}")

        print("\n" + "=" * 60)
        print("Demo completed!")
        print("=" * 60)
    else:
        print(f"\nNote: Checkpoint not found at {checkpoint_path}")
        print("To run this demo with real inference:")
        print("  1. Place a trained model checkpoint at test_data/best_model.pth")
        print("  2. Or modify checkpoint_path in this script")
        print("\nAPI call example (would work with valid checkpoint):")
        print("  heatmap = inference(input_data, checkpoint_path='your_model.pth')")


def demo_batch_inference():
    """Demonstrate batch inference with dummy data."""
    print("\n" + "=" * 60)
    print("Batch Inference Demo")
    print("=" * 60)

    # Create batch of 5 samples
    batch_data = np.random.rand(5, 2, 64, 64)  # Shape: (N, 2, 64, 64)
    print(f"Input batch shape: {batch_data.shape}")

    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / "test_data" / "best_model.pth"

    if checkpoint_path.exists():
        heatmaps = inference(batch_data, checkpoint_path=str(checkpoint_path))
        print(f"Output batch shape: {heatmaps.shape}")  # (5, 1, 64, 64)
        print("\nBatch inference completed!")
    else:
        print("\nNote: Checkpoint not found.")
        print("API call example:")
        print("  heatmaps = inference(batch_data, checkpoint_path='your_model.pth')")
        print(f"  # Output shape would be: ({batch_data.shape[0]}, 1, 64, 64)")


def main():
    """Main entry point - tries real data first, falls back to dummy."""
    project_root = Path(__file__).parent.parent
    test_data_dir = project_root / "test_data" / "01000165"
    checkpoint_path = project_root / "test_data" / "best_model.pth"

    # Check if real data is available
    has_real_data = test_data_dir.exists() and checkpoint_path.exists()

    if has_real_data:
        demo_with_real_data()
    else:
        print("CNN-CSF Library - Inference Demo")
        print("=" * 60)
        if not checkpoint_path.exists():
            print("\nCheckpoint not found. Using dummy data for API demonstration.")
            print("Place a trained model at test_data/best_model.pth for real inference.")
        demo_with_dummy_data()
        demo_batch_inference()


if __name__ == "__main__":
    main()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from model import LightweightUNet
from dataset import MedicalImageDataset


def load_model(model_path, device='cpu'):
    """Load trained model"""
    model = LightweightUNet(in_channels=2, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_heatmap(model, input1_path, input2_path, device='cpu'):
    """Predict heatmap for given inputs"""
    # Create temporary dataset for single prediction
    temp_data = pd.DataFrame({
        'input1': [input1_path],
        'input2': [input2_path],
        'output': [input1_path]  # Dummy output, not used for prediction
    })
    temp_data.to_csv('temp_data.list', sep=' ', index=False, header=False)
    
    # Load data
    dataset = MedicalImageDataset('temp_data.list', sigma=2.0, transform=None)
    inputs, _ = dataset[0]
    
    # Predict
    with torch.no_grad():
        inputs = inputs.unsqueeze(0).to(device)
        heatmap = model(inputs).squeeze().cpu().numpy()
    
    # Cleanup
    Path('temp_data.list').unlink()
    
    return heatmap


def extract_peaks(heatmap, threshold=0.5, min_distance=3, max_peaks=5):
    """
    Extract peak coordinates from heatmap
    
    Args:
        heatmap: 2D numpy array of predicted heatmap
        threshold: Minimum intensity threshold for peaks
        min_distance: Minimum distance between peaks
        max_peaks: Maximum number of peaks to return
    
    Returns:
        List of (x, y) coordinate tuples
    """
    from scipy.ndimage import maximum_filter
    
    # Apply maximum filter to find local maxima
    max_filtered = maximum_filter(heatmap, size=min_distance*2+1)
    maxima_mask = (heatmap == max_filtered) & (heatmap > threshold)
    
    # Get coordinates of maxima
    y_coords, x_coords = np.where(maxima_mask)
    intensities = heatmap[maxima_mask]
    
    if len(x_coords) == 0:
        return []
    
    # Sort by intensity (descending) and take top peaks
    sorted_indices = np.argsort(intensities)[::-1][:max_peaks]
    peaks = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
    
    return peaks


def visualize_results(input1_path, input2_path, heatmap, peaks, save_path=None):
    """Visualize input channels, predicted heatmap, and detected peaks"""
    # Load inputs
    def load_data(file_path):
        data = pd.read_csv(file_path, sep='\s+', header=None, names=['x', 'y', 'value'])
        grid = np.zeros((64, 64))
        for _, row in data.iterrows():
            x, y, value = int(row['x']), int(row['y']), float(row['value'])
            if 0 <= x < 64 and 0 <= y < 64:
                grid[y, x] = value
        return grid
    
    input1 = load_data(input1_path)
    input2 = load_data(input2_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Input channels
    axes[0, 0].imshow(input1, cmap='gray')
    axes[0, 0].set_title('Input Channel 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(input2, cmap='gray')
    axes[0, 1].set_title('Input Channel 2')
    axes[0, 1].axis('off')
    
    # Predicted heatmap
    im = axes[1, 0].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Combined visualization
    axes[1, 1].imshow(input1, cmap='gray', alpha=0.7)
    axes[1, 1].imshow(heatmap, cmap='hot', alpha=0.3, vmin=0, vmax=1)
    
    # Mark detected peaks
    if peaks:
        x_coords, y_coords = zip(*peaks)
        axes[1, 1].scatter(x_coords, y_coords, c='red', s=50, marker='x')
    
    axes[1, 1].set_title('Input with Peaks')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def batch_inference(model_path, data_list_path, output_dir, threshold=0.5, device='cpu'):
    """Run inference on all data in data.list"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model(model_path, device)
    
    # Load data list
    data_list = pd.read_csv(data_list_path, sep='\s+', header=None, names=['input1', 'input2', 'output'])
    
    results = []
    
    for idx, row in data_list.iterrows():
        print(f"Processing sample {idx+1}/{len(data_list)}...")
        
        # Predict heatmap
        heatmap = predict_heatmap(model, row['input1'], row['input2'], device)
        
        # Extract peaks
        peaks = extract_peaks(heatmap, threshold=threshold)
        
        # Save results
        result = {
            'sample_idx': idx,
            'input1': row['input1'],
            'input2': row['input2'],
            'output': row['output'],
            'peaks': peaks,
            'num_peaks': len(peaks)
        }
        results.append(result)
        
        # Visualize and save
        vis_path = output_dir / f'sample_{idx:03d}_results.png'
        visualize_results(row['input1'], row['input2'], heatmap, peaks, str(vis_path))
        
        # Save heatmap as numpy array
        np.save(output_dir / f'sample_{idx:03d}_heatmap.npy', heatmap)
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'inference_results.csv', index=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference for medical image point detection')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data_list', default='data.list', help='Path to data.list file')
    parser.add_argument('--input1', help='Path to first input file (single prediction)')
    parser.add_argument('--input2', help='Path to second input file (single prediction)')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Peak detection threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    if args.input1 and args.input2:
        # Single prediction
        print("Running single prediction...")
        model = load_model(args.model, device)
        heatmap = predict_heatmap(model, args.input1, args.input2, device)
        peaks = extract_peaks(heatmap, threshold=args.threshold)
        
        print(f"Detected {len(peaks)} peaks:")
        for i, (x, y) in enumerate(peaks):
            print(f"  Peak {i+1}: ({x}, {y})")
        
        visualize_results(args.input1, args.input2, heatmap, peaks)
        
    else:
        # Batch inference
        print("Running batch inference...")
        results = batch_inference(args.model, args.data_list, args.output_dir, 
                                threshold=args.threshold, device=device)
        
        print(f"\nInference completed! Results saved to: {args.output_dir}")
        print(f"Total samples processed: {len(results)}")
        print(f"Average peaks per sample: {np.mean([r['num_peaks'] for r in results]):.2f}")


if __name__ == '__main__':
    main()
"""Inference utilities for CNN-CSF library."""
import torch
import torch.nn as nn
import numpy as np


def extract_peaks(heatmap, threshold=0.5, min_distance=3):
    """
    Extract peak points from heatmap.

    Args:
        heatmap: Input heatmap tensor (1, 64, 64) or numpy array
        threshold: Minimum value threshold for peaks
        min_distance: Minimum distance between peaks

    Returns:
        List of (x, y) coordinates of peaks
    """
    # Convert to tensor if numpy array
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap).float()
        if heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(0)

    heatmap = heatmap.squeeze()
    peaks = []

    # Find local maxima
    max_pool = nn.MaxPool2d(min_distance*2+1, stride=1, padding=min_distance)
    maxima = (heatmap == max_pool(heatmap.unsqueeze(0).unsqueeze(0))).squeeze()

    # Filter by threshold
    mask = (heatmap > threshold) & maxima

    # Get coordinates
    if mask.any():
        y_coords, x_coords = torch.where(mask)
        for x, y in zip(x_coords, y_coords):
            peaks.append((x.item(), y.item()))

    return peaks


def run_inference(model, input_data, device='cpu'):
    """
    Run inference on input data.

    Args:
        model: Trained model
        input_data: Input numpy array (2, 64, 64) or batch (N, 2, 64, 64)
        device: Device to run inference on

    Returns:
        Output heatmap numpy array (1, 64, 64) or batch (N, 1, 64, 64)
    """
    model.eval()

    # Convert numpy to tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    else:
        input_tensor = input_data

    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Move to device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Convert back to numpy
    return output.cpu().numpy()

"""Dataset classes for CNN-CSF library."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import pickle


def _apply_gaussian_smoothing(sparse_label, sigma, device):
    """
    Apply Gaussian smoothing to sparse labels.

    Args:
        sparse_label: Sparse label tensor (1, 64, 64)
        sigma: Standard deviation for Gaussian kernel
        device: Target device for output tensor

    Returns:
        Heatmap tensor (1, 64, 64)
    """
    # Convert sparse label to numpy array
    sparse_np = sparse_label.squeeze().cpu().numpy()

    # Create Gaussian heatmap
    heatmap = np.zeros((64, 64))

    # Find all positive points
    positive_points = np.where(sparse_np > 0)

    if len(positive_points[0]) > 0:
        # Create meshgrid for Gaussian calculation
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))

        # Apply Gaussian smoothing to each positive point
        for y, x in zip(positive_points[0], positive_points[1]):
            # Create Gaussian kernel centered at (x, y)
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian)

    # Convert back to tensor and move to device
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).to(device)
    return heatmap_tensor


class MedicalImageDataset(Dataset):
    """
    Medical image dataset for point detection.

    Args:
        data_list_path: Path to data list file with format: input1_path, input2_path, output_path
        sigma: Standard deviation for Gaussian kernel in heatmap generation
        transform: Optional transforms to apply
        device: Target device to load data onto ('cpu', 'cuda', 'mps')
        cache_dir: Directory to store/load cached data
    """

    def __init__(self, data_list_path, sigma=2.0, transform=None, device='cpu', cache_dir='data_cache'):
        self.data_list = pd.read_csv(data_list_path, sep=',', header=None,
                                   names=['input1', 'input2', 'output'])
        self.sigma = sigma
        self.transform = transform
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)

        # Determine cache filename
        dataset_name = Path(data_list_path).stem
        self.cache_file = self.cache_dir / f"{dataset_name}.bin"

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

        # Try to load from cache
        if self.cache_file.exists():
            self.data_cache = self._load_cache()
        else:
            self.data_cache = self._load_from_original()
            self._save_cache()

    def __len__(self):
        return len(self.data_list)

    def _load_data(self, file_path):
        """Load 64x64 data from text file."""
        data = pd.read_csv(file_path, sep=r'\s+', header=None,
                          names=['x', 'y', 'value'])

        # Reshape to 64x64
        grid = np.zeros((64, 64))
        for _, row in data.iterrows():
            x, y, value = int(row['x']), int(row['y']), float(row['value'])
            if 0 <= x < 64 and 0 <= y < 64:
                grid[y, x] = value

        return grid.astype(np.float32)

    def _load_from_original(self):
        """Load data from original files (cache original labels, don't cache Gaussian smoothing)."""
        data_cache = []

        for idx in range(len(self.data_list)):
            row = self.data_list.iloc[idx]

            # Load input channels
            input1 = self._load_data(row['input1'])
            input2 = self._load_data(row['input2'])

            # Stack inputs to create 2-channel input
            inputs = np.stack([input1, input2], axis=0)

            # Load original label points
            label_data = pd.read_csv(row['output'], sep=r'\s+', header=None,
                                   names=['x', 'y', 'value'])
            # Create sparse label: 0/1 matrix
            sparse_label = np.zeros((64, 64))
            for _, point in label_data.iterrows():
                x, y, value = int(point['x']), int(point['y']), float(point['value'])
                if 0 <= x < 64 and 0 <= y < 64 and value > 0:
                    sparse_label[y, x] = 1.0

            # Convert to torch tensors and move to device
            inputs = torch.from_numpy(inputs).float().to(self.device)
            sparse_label = torch.from_numpy(sparse_label).float().unsqueeze(0).to(self.device)

            data_cache.append((inputs, sparse_label))

        return data_cache

    def _save_cache(self):
        """Save data to cache file."""
        try:
            # Move data to CPU before saving
            cpu_data = []
            for inputs, target in self.data_cache:
                cpu_data.append((inputs.cpu(), target.cpu()))

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cpu_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_cache(self):
        """Load data from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                cpu_data = pickle.load(f)

            # Move data to target device
            device_data = []
            for inputs, target in cpu_data:
                device_data.append((inputs.to(self.device), target.to(self.device)))

            return device_data
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            print("Falling back to original file loading...")
            return self._load_from_original()

    def __getitem__(self, idx):
        inputs, sparse_label = self.data_cache[idx]

        # Apply Gaussian smoothing at runtime
        target = _apply_gaussian_smoothing(sparse_label, self.sigma, self.device)

        # If data augmentation is needed, apply transform here
        if self.transform:
            inputs = self.transform(inputs.clone())

        return inputs, target


class RandomTransform:
    """Simple data augmentation for 64x64 images."""

    def __init__(self, shift_range=3, scale_range=0.1):
        self.shift_range = shift_range
        self.scale_range = scale_range

    def __call__(self, image):
        """Apply random transformations to 2-channel image."""
        # For now, skip transforms to avoid complications
        return image

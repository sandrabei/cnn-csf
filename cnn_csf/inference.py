"""Inference API for CNN-CSF library."""
import torch
import numpy as np
from pathlib import Path
from .model import LightweightUNet
from .utils import setup_device


def inference(input_data, checkpoint_path, device='auto'):
    """
    Run inference on input data using a trained model.

    This is the main inference API for the CNN-CSF library. It takes a numpy array
    as input (2-channel 64x64 medical image) and returns the predicted heatmap.

    Args:
        input_data: Input numpy array with shape (2, 64, 64) or (N, 2, 64, 64)
                   The two channels should be EPI and T1 data respectively.
                   Values should be normalized to 0-1 range.
        checkpoint_path: Path to the model checkpoint file (.pth)
        device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        numpy.ndarray: Predicted heatmap with shape (1, 64, 64) or (N, 1, 64, 64)
                      Values are in range 0-1, representing point probability.

    Raises:
        ValueError: If input shape is incorrect
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> import numpy as np
        >>> from cnn_csf import inference
        >>>
        >>> # Load or prepare your data (2 channels: EPI and T1)
        >>> epi_data = np.random.rand(64, 64)  # Your EPI data
        >>> t1_data = np.random.rand(64, 64)   # Your T1 data
        >>> input_data = np.stack([epi_data, t1_data], axis=0)  # Shape: (2, 64, 64)
        >>>
        >>> # Run inference
        >>> heatmap = inference(input_data, checkpoint_path='path/to/model.pth')
        >>> print(heatmap.shape)  # (1, 64, 64)

    Note:
        - The input should be preprocessed and normalized to 0-1 range before passing
        - For 3D volumes, process each slice separately through this function
        - The output heatmap can be used to extract point locations using peak detection
    """
    # Check input shape
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 2:
            raise ValueError(f"Input must have shape (2, 64, 64) or (N, 2, 64, 64), got {input_data.shape}")
        if input_data.ndim == 3 and input_data.shape[0] != 2:
            raise ValueError(f"First dimension must be 2 (for EPI and T1 channels), got {input_data.shape[0]}")
        if input_data.shape[-2:] != (64, 64):
            raise ValueError(f"Spatial dimensions must be 64x64, got {input_data.shape[-2:]}")
    elif isinstance(input_data, torch.Tensor):
        if input_data.ndim == 2:
            raise ValueError(f"Input must have shape (2, 64, 64) or (N, 2, 64, 64), got {input_data.shape}")
        if input_data.ndim == 3 and input_data.shape[0] != 2:
            raise ValueError(f"First dimension must be 2 (for EPI and T1 channels), got {input_data.shape[0]}")
        if input_data.shape[-2:] != (64, 64):
            raise ValueError(f"Spatial dimensions must be 64x64, got {input_data.shape[-2:]}")

    # Check checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Setup device
    device = setup_device(device)

    # Load model
    model = LightweightUNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Prepare input
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    else:
        input_tensor = input_data.float()

    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Move to device
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert to numpy
    return output.cpu().numpy()

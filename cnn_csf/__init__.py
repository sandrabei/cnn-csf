"""
CNN-CSF: A library for 3D medical image point detection.

This library provides tools for:
- Running inference on medical images using trained models
- Fine-tuning models on custom datasets

Main APIs:
    inference: Run model inference on numpy arrays
    finetune: Fine-tune the model on custom data
"""

from .inference import inference
from .finetune import finetune
from .model import LightweightUNet
from .loss import get_loss_function, list_loss_functions

__version__ = "0.1.0"

__all__ = [
    'inference',
    'finetune',
    'LightweightUNet',
    'get_loss_function',
    'list_loss_functions',
]

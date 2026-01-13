"""Loss functions module for CNN-CSF library."""
from .losses import (
    GeneralizedCrossEntropyLoss,
    FocalLoss,
    get_loss_function,
    list_loss_functions
)

__all__ = [
    'GeneralizedCrossEntropyLoss',
    'FocalLoss',
    'get_loss_function',
    'list_loss_functions'
]

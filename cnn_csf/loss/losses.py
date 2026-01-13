"""Loss function implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy Loss

    Suitable for robust learning with noisy labels.

    Args:
        q (float): GCE hyperparameter, controls the shape of the loss function. 0 < q <= 1
                  When q approaches 0, it approaches MAE; when q=1, it's standard BCE
        reduction (str): Loss aggregation method, options: 'mean', 'sum', 'none'

    Shape:
        - Input: (N, *) Model prediction probabilities
        - Target: (N, *) True labels (0 or 1)
        - Output: If reduction='none', same shape as input; otherwise scalar
    """

    def __init__(self, q=0.4, reduction='mean'):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        assert 0 < q <= 1, "q must be in (0, 1]"
        self.q = q
        self.reduction = reduction

    def forward(self, input, target):
        # Ensure input is in [0,1] range
        input = torch.clamp(input, min=1e-7, max=1-1e-7)

        # Calculate pt: model's prediction probability for correct label
        pt = torch.where(target == 1, input, 1 - input)

        # GCE loss function: (1 - pt^q) / q
        if self.q == 1.0:
            loss = -torch.log(pt)
        else:
            loss = (1 - torch.pow(pt, self.q)) / self.q

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection

    Loss function that focuses on hard samples by adjusting gamma and alpha parameters.

    Args:
        alpha (float): Weight to balance positive/negative samples, default 0.25
        gamma (float): Focusing parameter, controls attention to hard samples, default 2.0
        reduction (str): Loss aggregation method, options: 'mean', 'sum', 'none'

    Shape:
        - Input: (N, *) Model prediction probabilities
        - Target: (N, *) True labels (0 or 1)
        - Output: If reduction='none', same shape as input; otherwise scalar
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Ensure input is in [0,1] range
        input = torch.clamp(input, min=1e-7, max=1-1e-7)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')

        # Calculate pt
        pt = torch.where(target == 1, input, 1 - input)

        # Focal Loss: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * torch.pow(1 - pt, self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def get_loss_function(loss_type='gce', **kwargs):
    """
    Factory function for obtaining loss functions

    Args:
        loss_type (str): Loss function type, options: 'gce', 'bce', 'mae', 'mse', 'focal'
        **kwargs: Parameters passed to specific loss function

    Returns:
        nn.Module: Loss function instance
    """
    loss_type = loss_type.lower()

    if loss_type == 'gce':
        q = kwargs.get('gce_q', 0.4)
        return GeneralizedCrossEntropyLoss(q=q)
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def list_loss_functions():
    """Return list of supported loss functions"""
    return ['gce', 'bce', 'mae', 'mse', 'focal']

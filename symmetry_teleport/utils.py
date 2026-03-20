"""
Utility functions for symmetry teleportation.
"""

import torch


def get_param_norm(model):
    """
    Compute L2 norm of all model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: L2 norm of flattened parameters
    """
    with torch.no_grad():
        total = 0.0
        for p in model.parameters():
            total += (p.detach().float() ** 2).sum().item()
        return total ** 0.5


def get_grad_norm(model):
    """
    Compute L2 norm of all gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: L2 norm of flattened gradients
    """
    with torch.no_grad():
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.detach() ** 2).sum().item()
        return total ** 0.5


def count_parameters(model):
    """
    Count total number of parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

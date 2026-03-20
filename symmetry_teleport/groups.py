"""
Symmetry group implementations for teleportation.

This module provides symmetry transformations that can be used with TeleportSGD.
"""

import torch
import torch.nn as nn


class ScalarRescalingGroup:
    """
    Diagonal scaling symmetry for feedforward networks with ReLU/GELU activation.
    
    For a two-layer FFN: y = W2 @ activation(W1 @ x + b1) + b2
    
    The transformation g(s) with s > 0 (size = hidden_dim):
        W1' = diag(s) @ W1  (row-wise scaling)
        b1' = s * b1        (element-wise)
        W2' = W2 @ diag(1/s)  (column-wise scaling)
        b2' = b2            (unchanged)
    
    This preserves the function output for ReLU/GELU activations.
    
    Example:
        >>> group = ScalarRescalingGroup()
        >>> # Get layer parameters
        >>> linear1, linear2 = model.get_ffn_layers(layer_idx=0)
        >>> # Apply scaling
        >>> s_vec = torch.exp(torch.randn(hidden_dim) * 0.1)
        >>> group.apply_transform(linear1, linear2, s_vec)
    """
    
    @staticmethod
    def apply_transform(linear1: nn.Linear, linear2: nn.Linear, s_vec: torch.Tensor):
        """
        Apply diagonal scaling transformation in-place.
        
        Args:
            linear1: First linear layer (d_model -> hidden_dim)
            linear2: Second linear layer (hidden_dim -> d_model)
            s_vec: Scaling factors (size = hidden_dim), must be positive
        """
        with torch.no_grad():
            s = s_vec.to(linear1.weight.device, linear1.weight.dtype)
            
            # Scale rows of W1 and entries of b1
            linear1.weight.mul_(s[:, None])
            if linear1.bias is not None:
                linear1.bias.mul_(s)
            
            # Scale columns of W2 by 1/s
            linear2.weight.mul_(1.0 / s[None, :])
            # linear2.bias unchanged (if exists)
    
    @staticmethod
    def get_identity(hidden_dim: int, device=None, dtype=None) -> torch.Tensor:
        """
        Get identity transformation (s = 1 for all dimensions).
        
        Args:
            hidden_dim: Size of hidden dimension
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tensor of ones with shape (hidden_dim,)
        """
        return torch.ones(hidden_dim, device=device, dtype=dtype)
    
    @staticmethod
    def validate_transform(s_vec: torch.Tensor) -> bool:
        """
        Check if transformation is valid (all positive).
        
        Args:
            s_vec: Scaling factors
            
        Returns:
            True if all elements are positive
        """
        return (s_vec > 0).all().item()

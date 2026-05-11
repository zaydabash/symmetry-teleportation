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


class AttentionQKGroup:
    """
    Q/K projection symmetry for self-attention layers.

    Derivation
    ----------
    PyTorch nn.Linear convention: output = x @ W.T + b, W shape (out_features, in_features).

    For nn.MultiheadAttention with qkv_same_embed_dim=True, projections are packed in
    in_proj_weight of shape (3*d, d_model) and in_proj_bias of shape (3*d,):

        W_Q = in_proj_weight[0:d,   :]     Q = X @ W_Q.T + b_Q
        W_K = in_proj_weight[d:2*d, :]     K = X @ W_K.T + b_K

    Attention logits: Attn = Q @ K.T / sqrt(d)

    Symmetry: for invertible A in GL(d), define Q' = Q @ A,  K' = K @ A^{-T}.

    Invariance:
        Q' @ K'.T = (Q @ A) @ (K @ A^{-T}).T
                  = Q @ A @ A^{-1} @ K.T
                  = Q @ K.T   (exact)

    The 1/sqrt(d) scaling is a constant multiplier on both sides and cancels.

    Weight-space transforms (left-multiply on weight rows, right-multiply on bias):

        W_Q' = A.T @ W_Q      because X @ (A.T @ W_Q).T = X @ W_Q.T @ A = Q @ A
        W_K' = A_inv @ W_K    because X @ (A_inv @ W_K).T = X @ W_K.T @ A_inv.T = K @ A^{-T}
        b_Q' = b_Q @ A        bias shifts Q by b_Q, so Q' gets b_Q @ A
        b_K' = b_K @ A_inv.T  bias shifts K by b_K, so K' gets b_K @ A^{-T}

    Multi-head constraint
    ---------------------
    For nhead > 1, PyTorch splits Q and K into nhead slices of size d_h = d / nhead.
    Head h receives Q_h = Q[:, :, h*d_h:(h+1)*d_h].

    Per-head logit Q_h @ K_h.T is preserved exactly only when A is block-diagonal:

        A = diag(A_0, A_1, ..., A_{nhead-1}),  each A_i in GL(d_h)

    For nhead = 1, any invertible A in GL(d) preserves the full attention output.
    Use make_block_diagonal() to assemble A from per-head matrices.
    """

    @staticmethod
    def apply_transform(attn: nn.MultiheadAttention, A: torch.Tensor) -> None:
        """
        Apply Q/K symmetry transform in-place to nn.MultiheadAttention.

        Modifies in_proj_weight (and in_proj_bias if present) so that the
        attention logits Q @ K.T are preserved up to floating-point rounding.

        Args:
            attn: nn.MultiheadAttention with qkv_same_embed_dim=True (packed
                  in_proj_weight). Cross-attention with separate q/k/v projections
                  (in_proj_weight is None) is not supported.
            A:    Invertible matrix of shape (d, d) where d = attn.embed_dim.
                  For nhead > 1: must be block-diagonal with nhead blocks of size
                  (d_h, d_h) to preserve per-head attention outputs.
                  For nhead = 1: any invertible A preserves the attention output.

        Raises:
            ValueError: if attn.in_proj_weight is None.
        """
        if attn.in_proj_weight is None:
            raise ValueError(
                "AttentionQKGroup.apply_transform requires in_proj_weight "
                "(qkv_same_embed_dim=True). Separate q/k/v projections are not supported."
            )

        d = attn.embed_dim
        A = A.to(device=attn.in_proj_weight.device, dtype=attn.in_proj_weight.dtype)
        A_inv = torch.linalg.inv(A)

        with torch.no_grad():
            W = attn.in_proj_weight  # (3*d, d_model), Parameter
            # Clone slices before in-place assignment to avoid aliasing.
            W_Q_new = A.T @ W[0:d, :].clone()    # A.T @ W_Q
            W_K_new = A_inv @ W[d:2 * d, :].clone()  # A_inv @ W_K
            W[0:d, :].copy_(W_Q_new)
            W[d:2 * d, :].copy_(W_K_new)

            if attn.in_proj_bias is not None:
                b = attn.in_proj_bias  # (3*d,), Parameter
                b_Q_new = b[0:d].clone() @ A          # b_Q @ A
                b_K_new = b[d:2 * d].clone() @ A_inv.T  # b_K @ A^{-T}
                b[0:d].copy_(b_Q_new)
                b[d:2 * d].copy_(b_K_new)

    @staticmethod
    def make_block_diagonal(
        head_matrices: list,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Build a block-diagonal matrix from per-head invertible matrices.

        The resulting matrix A = diag(A_0, ..., A_{nhead-1}) satisfies the
        block-diagonal constraint required for multi-head attention invariance.

        Args:
            head_matrices: List of nhead tensors, each of shape (d_h, d_h).
            device: Target device (defaults to first matrix's device).
            dtype:  Target dtype  (defaults to first matrix's dtype).

        Returns:
            Block-diagonal tensor of shape (nhead * d_h, nhead * d_h).
        """
        device = device if device is not None else head_matrices[0].device
        dtype = dtype if dtype is not None else head_matrices[0].dtype
        mats = [m.to(device=device, dtype=dtype) for m in head_matrices]
        return torch.block_diag(*mats)

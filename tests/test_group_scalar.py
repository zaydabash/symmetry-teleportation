"""
Tests for ScalarRescalingGroup symmetry transformation.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symmetry_teleport import ScalarRescalingGroup


def test_apply_transform_and_inverse():
    """Test that applying transform and inverse returns original parameters."""
    torch.manual_seed(42)
    
    # Create two linear layers (FFN)
    hidden_dim = 64
    d_model = 32
    
    linear1 = nn.Linear(d_model, hidden_dim)
    linear2 = nn.Linear(hidden_dim, d_model)
    
    # Save original parameters
    w1_orig = linear1.weight.clone()
    b1_orig = linear1.bias.clone()
    w2_orig = linear2.weight.clone()
    b2_orig = linear2.bias.clone()
    
    # Apply random scaling
    s_vec = torch.exp(torch.randn(hidden_dim) * 0.5)
    ScalarRescalingGroup.apply_transform(linear1, linear2, s_vec)
    
    # Apply inverse scaling
    s_vec_inv = 1.0 / s_vec
    ScalarRescalingGroup.apply_transform(linear1, linear2, s_vec_inv)
    
    # Check that parameters are restored (within tolerance)
    assert torch.allclose(linear1.weight, w1_orig, rtol=1e-5, atol=1e-6)
    assert torch.allclose(linear1.bias, b1_orig, rtol=1e-5, atol=1e-6)
    assert torch.allclose(linear2.weight, w2_orig, rtol=1e-5, atol=1e-6)
    assert torch.allclose(linear2.bias, b2_orig, rtol=1e-5, atol=1e-6)


def test_identity_transform():
    """Test that identity transformation (s=1) leaves parameters unchanged."""
    torch.manual_seed(42)
    
    hidden_dim = 64
    d_model = 32
    
    linear1 = nn.Linear(d_model, hidden_dim)
    linear2 = nn.Linear(hidden_dim, d_model)
    
    # Save original parameters
    w1_orig = linear1.weight.clone()
    b1_orig = linear1.bias.clone()
    w2_orig = linear2.weight.clone()
    b2_orig = linear2.bias.clone()
    
    # Apply identity transformation
    s_vec = ScalarRescalingGroup.get_identity(hidden_dim)
    ScalarRescalingGroup.apply_transform(linear1, linear2, s_vec)
    
    # Check that parameters are unchanged
    assert torch.allclose(linear1.weight, w1_orig)
    assert torch.allclose(linear1.bias, b1_orig)
    assert torch.allclose(linear2.weight, w2_orig)
    assert torch.allclose(linear2.bias, b2_orig)


def test_validate_transform():
    """Test validation of transformation vectors."""
    hidden_dim = 64
    
    # Valid: all positive
    s_valid = torch.ones(hidden_dim) * 2.0
    assert ScalarRescalingGroup.validate_transform(s_valid)
    
    # Invalid: contains zero
    s_zero = torch.ones(hidden_dim)
    s_zero[0] = 0.0
    assert not ScalarRescalingGroup.validate_transform(s_zero)
    
    # Invalid: contains negative
    s_neg = torch.ones(hidden_dim)
    s_neg[0] = -1.0
    assert not ScalarRescalingGroup.validate_transform(s_neg)


def test_function_invariance_relu():
    """Test that transformation preserves function output for ReLU activation."""
    torch.manual_seed(42)
    
    hidden_dim = 64
    d_model = 32
    batch_size = 8
    
    # Create FFN with ReLU
    linear1 = nn.Linear(d_model, hidden_dim)
    linear2 = nn.Linear(hidden_dim, d_model)
    
    def ffn(x):
        return linear2(torch.relu(linear1(x)))
    
    # Generate random input
    x = torch.randn(batch_size, d_model)
    
    # Output before transformation
    out_before = ffn(x)
    
    # Apply random scaling
    s_vec = torch.exp(torch.randn(hidden_dim) * 0.3)
    ScalarRescalingGroup.apply_transform(linear1, linear2, s_vec)
    
    # Output after transformation
    out_after = ffn(x)
    
    # Outputs should be identical (within numerical tolerance)
    assert torch.allclose(out_before, out_after, rtol=1e-5, atol=1e-6), \
        f"Max diff: {(out_before - out_after).abs().max().item()}"


def test_get_identity():
    """Test get_identity returns correct tensor."""
    hidden_dim = 64
    
    s_identity = ScalarRescalingGroup.get_identity(hidden_dim)
    
    assert s_identity.shape == (hidden_dim,)
    assert torch.allclose(s_identity, torch.ones(hidden_dim))
    
    # Test with device and dtype
    if torch.cuda.is_available():
        s_cuda = ScalarRescalingGroup.get_identity(hidden_dim, device='cuda', dtype=torch.float64)
        assert s_cuda.device.type == 'cuda'
        assert s_cuda.dtype == torch.float64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

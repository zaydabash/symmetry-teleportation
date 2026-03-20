"""
Tests for new teleportation features:
1. Direct S parameterization (softplus)
2. Grad norm objective
3. Before/after invariance logging
"""

import pytest
import torch
import torch.nn as nn
from symmetry_teleport.teleport import sym_param_to_s_vec, objective_virtual_loss, objective_grad_norm, teleport_ffn_diagonal
from symmetry_teleport import TeleportSGD
from transformer_teleport_optimizer import TinyTransformer


def test_sym_param_to_s_vec_exp():
    """Test exp parameterization: s = exp(log_s)"""
    log_s = torch.tensor([0.0, 1.0, -1.0])
    s_vec = sym_param_to_s_vec(log_s, s_param='exp')
    
    expected = torch.exp(log_s)
    assert torch.allclose(s_vec, expected)
    assert (s_vec > 0).all()


def test_sym_param_to_s_vec_direct():
    """Test direct parameterization: s = softplus(u) + eps"""
    u = torch.tensor([0.0, 1.0, -1.0])
    s_vec = sym_param_to_s_vec(u, s_param='direct')
    
    # Check all positive
    assert (s_vec > 0).all()
    
    # Check reasonable range (softplus(0) ≈ 0.69, softplus(1) ≈ 1.31, softplus(-1) ≈ 0.31)
    assert s_vec[0] > 0.6 and s_vec[0] < 0.8
    assert s_vec[1] > 1.2
    assert s_vec[2] > 0.2 and s_vec[2] < 0.4


def test_sym_param_to_s_vec_projected():
    """Test projected parameterization: s directly (already clamped)"""
    s = torch.tensor([1.0, 2.0, 0.5])
    s_vec = sym_param_to_s_vec(s, s_param='projected')
    
    # Should return s as-is
    assert torch.allclose(s_vec, s)
    assert (s_vec > 0).all()


def test_sym_param_to_s_vec_clamping():
    """Test that exp parameterization respects log_s_clip"""
    log_s = torch.tensor([0.0, 5.0, -5.0])  # Out of bounds
    s_vec = sym_param_to_s_vec(log_s, s_param='exp', log_s_clip=(-2.0, 2.0))
    
    # Should be clamped: exp(-2), exp(2), exp(-2)
    expected = torch.tensor([1.0, torch.exp(torch.tensor(2.0)), torch.exp(torch.tensor(-2.0))])
    assert torch.allclose(s_vec, expected, atol=1e-5)


def test_objective_virtual_loss_shape():
    """Test virtual_loss objective returns scalar with gradients"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    model.eval()
    
    X = torch.randn(4, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(4, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    # Create parameters with grad
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
    from collections import OrderedDict
    params_ordered = OrderedDict(params)
    
    sym_param = torch.zeros(16, requires_grad=True)
    
    J = objective_virtual_loss(model, params_ordered, sym_param, X, Y, loss_fn, lr=0.01, lambda_penalty=1e-3)
    
    # Check shape and grad_fn
    assert J.shape == torch.Size([])
    assert J.grad_fn is not None
    
    # Check gradient can be computed
    J.backward()
    assert sym_param.grad is not None


def test_objective_grad_norm_shape():
    """Test grad_norm objective returns scalar with gradients"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    model.eval()
    
    X = torch.randn(4, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(4, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    # Create parameters with grad
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
    from collections import OrderedDict
    params_ordered = OrderedDict(params)
    
    sym_param = torch.zeros(16, requires_grad=True)
    
    J = objective_grad_norm(model, params_ordered, sym_param, X, Y, loss_fn, lambda_penalty=1e-3)
    
    # Check shape and grad_fn
    assert J.shape == torch.Size([])
    assert J.grad_fn is not None
    
    # Check gradient can be computed
    J.backward()
    assert sym_param.grad is not None


def test_teleport_ffn_diagonal_exp_virtual_loss():
    """Test teleportation with exp param and virtual_loss objective"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    model.eval()
    
    X = torch.randn(8, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(8, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    s_best, J_before, J_best, diagnostics = teleport_ffn_diagonal(
        model, 0, X, loss_fn, Y,
        lr_theta=0.1, steps=5,
        objective='virtual_loss',
        s_param='exp'
    )
    
    # Check outputs
    assert s_best.shape == torch.Size([16])
    assert (s_best > 0).all()
    assert isinstance(J_before, float)
    assert isinstance(J_best, float)
    
    # Check diagnostics include new fields
    assert 'loss_before' in diagnostics
    assert 'loss_after' in diagnostics
    assert 'delta_loss' in diagnostics
    assert 'grad_norm_before' in diagnostics
    assert 'grad_norm_after' in diagnostics
    assert 'delta_grad_norm' in diagnostics
    
    # Check loss invariance (should be very small)
    assert abs(diagnostics['delta_loss']) < 1e-4


def test_teleport_ffn_diagonal_direct_grad_norm():
    """Test teleportation with direct param and grad_norm objective"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    model.eval()
    
    X = torch.randn(8, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(8, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    s_best, J_before, J_best, diagnostics = teleport_ffn_diagonal(
        model, 0, X, loss_fn, Y,
        lr_theta=0.1, steps=5,
        objective='grad_norm',
        s_param='direct'
    )
    
    # Check outputs
    assert s_best.shape == torch.Size([16])
    assert (s_best > 0).all()
    assert isinstance(J_before, float)
    assert isinstance(J_best, float)
    
    # Check diagnostics include new fields
    assert 'loss_before' in diagnostics
    assert 'loss_after' in diagnostics
    assert 'delta_loss' in diagnostics
    assert 'grad_norm_before' in diagnostics
    assert 'grad_norm_after' in diagnostics
    assert 'delta_grad_norm' in diagnostics
    
    # Check loss invariance
    assert abs(diagnostics['delta_loss']) < 1e-4


def test_teleport_sgd_with_new_options():
    """Test TeleportSGD optimizer with new objective and s_param options"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    
    X = torch.randn(8, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(8, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    optimizer = TeleportSGD(
        model.parameters(),
        lr=0.01,
        teleport_every=2,
        teleport_config={
            'model': model,
            'layer_idx': 0,
            'X_teleport': X,
            'Y_teleport': Y,
            'loss_fn': loss_fn,
            'lr_theta': 0.1,
            'inner_steps': 3,
            'objective': 'grad_norm',
            's_param': 'direct'
        }
    )
    
    # Run a few steps
    model.train()
    for i in range(4):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
    
    # Check that teleportation happened (2 attempts at steps 2 and 4)
    assert len(optimizer.teleport_attempts) == 2
    
    # Check that attempts have the new fields
    attempt = optimizer.teleport_attempts[0]
    assert 'loss_before_teleport' in attempt
    assert 'loss_after_teleport' in attempt
    assert 'delta_loss_teleport' in attempt
    assert 'grad_norm_before_teleport' in attempt
    assert 'grad_norm_after_teleport' in attempt
    assert 'delta_grad_norm_teleport' in attempt


def test_loss_invariance_property():
    """Test that diagonal scaling truly preserves loss (symmetry property)"""
    model = TinyTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    model.eval()
    
    X = torch.randn(8, 4, 4)  # (batch, seq_len, features)
    Y = torch.randn(8, 4)  # (batch, features)
    loss_fn = nn.MSELoss()
    
    # Compute loss before
    with torch.no_grad():
        out_before = model(X)
        loss_before = loss_fn(out_before, Y)
    
    # Apply a random diagonal scaling
    linear1, linear2 = model.get_ffn_layers(0)
    s_vec = torch.exp(torch.randn(16) * 0.5)
    
    with torch.no_grad():
        linear1.weight.mul_(s_vec[:, None])
        if linear1.bias is not None:
            linear1.bias.mul_(s_vec)
        linear2.weight.mul_(1.0 / s_vec[None, :])
    
    # Compute loss after
    with torch.no_grad():
        out_after = model(X)
        loss_after = loss_fn(out_after, Y)
    
    # Check invariance
    assert torch.allclose(loss_before, loss_after, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Basic tests for TeleportSGD optimizer.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symmetry_teleport import TeleportSGD


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


def test_optimizer_step_changes_parameters():
    """Test that optimizer step changes parameters."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = TeleportSGD(model.parameters(), lr=0.01, teleport_every=0)
    
    # Save initial parameters
    initial_params = {name: p.clone() for name, p in model.named_parameters()}
    
    # Forward pass and backward
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Check that parameters changed
    for name, p in model.named_parameters():
        assert not torch.allclose(p, initial_params[name]), f"Parameter {name} did not change"


def test_optimizer_zero_grad():
    """Test that zero_grad clears gradients."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = TeleportSGD(model.parameters(), lr=0.01, teleport_every=0)
    
    # Forward pass and backward
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    # Check gradients exist
    for p in model.parameters():
        assert p.grad is not None
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Check gradients are None or zero
    for p in model.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_optimizer_without_teleportation():
    """Test that optimizer works correctly without teleportation."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = TeleportSGD(model.parameters(), lr=0.1, teleport_every=0)
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    # Initial loss
    loss_before = nn.MSELoss()(model(x), y).item()
    
    # Train for a few steps
    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    # Final loss
    loss_after = nn.MSELoss()(model(x), y).item()
    
    # Loss should decrease
    assert loss_after < loss_before, "Loss did not decrease"


def test_optimizer_step_count():
    """Test that step count increments correctly."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = TeleportSGD(model.parameters(), lr=0.01, teleport_every=0)
    
    assert optimizer.step_count == 0
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    for i in range(5):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        assert optimizer.step_count == i + 1


def test_teleport_stats_empty():
    """Test that teleport_stats returns correct values when no teleportation occurred."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = TeleportSGD(model.parameters(), lr=0.01, teleport_every=0)
    
    stats = optimizer.teleport_stats()
    
    assert stats['total_attempts'] == 0
    assert stats['accepted_count'] == 0
    assert stats['active_count'] == 0
    assert stats['acceptance_rate'] == 0.0
    assert stats['active_rate_strict'] == 0.0
    assert len(stats['recent_delta_J']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

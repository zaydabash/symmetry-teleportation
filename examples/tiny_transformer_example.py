#!/usr/bin/env python3
"""
Minimal example: TeleportSGD on a tiny transformer.

Demonstrates how to plug TeleportSGD into a standard training loop.
The model must implement get_ffn_layers(layer_idx) returning
(linear1, linear2).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from symmetry_teleport import TeleportSGD  # noqa: E402


class SmallTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, dim_feedforward=32):
        super().__init__()
        self.embed = nn.Linear(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 4)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

    def get_ffn_layers(self, layer_idx=0):
        enc = self.encoder.layers[layer_idx]
        return enc.linear1, enc.linear2


torch.manual_seed(0)
X = torch.randn(128, 8, 4)
Y = X.mean(dim=(1, 2)).unsqueeze(1).expand(-1, 4)

model = SmallTransformer()
loss_fn = nn.MSELoss()

optimizer = TeleportSGD(
    model.parameters(),
    lr=0.01,
    teleport_every=5,
    teleport_config={
        "model": model,
        "layer_idx": 0,
        "X_teleport": X,
        "Y_teleport": Y,
        "loss_fn": loss_fn,
        "lr_theta": 0.3,
        "inner_steps": 50,
        "objective": "virtual_sgd_improve",
        "s_param": "projected",
        "log_s_clip": (-2.0, 2.0),
    },
)

print("step  loss")
for step in range(100):
    indices = torch.arange(step * 16, (step + 1) * 16) % len(X)
    optimizer.zero_grad()
    loss = loss_fn(model(X[indices]), Y[indices])
    loss.backward()
    optimizer.step()
    if step % 20 == 0 or step == 99:
        print(f"{step:>4}  {loss.item():.6f}")

stats = optimizer.teleport_stats()
print(f"\nTeleport attempts : {stats['total_attempts']}")
print(f"Accepted          : {stats['accepted_count']} ({stats['acceptance_rate']:.1f}%)")

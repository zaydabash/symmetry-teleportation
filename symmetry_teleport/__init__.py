"""
Symmetry Teleportation Optimizer

A PyTorch optimizer that exploits loss-invariant symmetries to potentially
improve convergence speed. Based on "Symmetry Teleportation for Accelerated
Optimization" (Zhao et al., NeurIPS 2022).

Example:
    >>> from symmetry_teleport import TeleportSGD, ScalarRescalingGroup
    >>> import torch.nn as nn
    >>> 
    >>> model = MyModel()
    >>> optimizer = TeleportSGD(
    ...     model.parameters(),
    ...     lr=0.01,
    ...     teleport_every=5,
    ...     teleport_config={
    ...         'model': model,
    ...         'X_teleport': X_batch,
    ...         'Y_teleport': Y_batch,
    ...         'loss_fn': nn.MSELoss()
    ...     }
    ... )
"""

__version__ = '0.1.0'

from .optim import TeleportSGD
from .groups import ScalarRescalingGroup

__all__ = ['TeleportSGD', 'ScalarRescalingGroup']

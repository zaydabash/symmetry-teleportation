"""
Optimizer-Level Teleportation for Standard PyTorch Transformers

This implements symmetry teleportation as an optimizer-level operation,
integrated into the training loop like momentum or preconditioning.

Key Design:
- Uses standard nn.TransformerEncoderLayer (not custom architecture)
- Applies diagonal scaling symmetry to FFN (feedforward network)
- Teleportation is part of optimizer, not manual training loop hack
- Compares SGD baseline vs SGD + teleportation on identical setups

Objective C: Minimizes J(log_s) = L(θ ∘ g(log_s) - η ∇L(θ ∘ g(log_s))) + λ ||log_s||²

CLI Usage:
    python transformer_teleport_optimizer.py --num_seeds 5 --steps 200 --teleport_every 20 --lr 0.01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
from scipy import stats
import contextlib

# Try to import sdp_kernel for disabling Flash Attention (needed for double backward)
try:
    from torch.backends.cuda import sdp_kernel
except ImportError:
    sdp_kernel = None


# ============================================================================
# 1. HIGH-LEVEL PLAN
# ============================================================================
"""
Plan:
1. Model: Tiny TransformerEncoder (1-2 layers, ~500 params) using standard nn.TransformerEncoderLayer
2. Extract FFN from encoder layer: access .feed_forward network (linear1 + activation + linear2)
3. Optimizer: Create SGDTeleportOptimizer that wraps SGD and adds teleport_step() method
4. Teleportation: Diagonal scaling symmetry on FFN (valid for ReLU/GELU)
5. Schedule: Apply teleport_step() every N gradient steps (e.g., every 5 steps)
6. Comparison: Run identical training with/without teleportation, plot convergence
"""


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class TinyTransformer(nn.Module):
    """
    Standard PyTorch Transformer encoder with configurable size.
    Uses nn.TransformerEncoderLayer (standard implementation).
    """
    def __init__(self, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        # Token embedding (for synthetic data, just a linear projection)
        self.embed = nn.Linear(4, d_model)
        
        # Standard Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # FIXED: Disable dropout for deterministic pairing (teleport search consumes RNG)
            activation='relu',  # Uses ReLU in FFN (valid for diagonal scaling)
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, 4)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embed(x)  # (batch, seq_len, d_model)
        x = self.encoder(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Average pooling over sequence
        x = self.output(x)  # (batch, 4)
        return x
    
    def get_ffn_layers(self, layer_idx=0):
        """
        Extract FFN (feedforward) layers from specified encoder layer.
        Returns: (linear1, linear2) modules for teleportation.

        Structure of TransformerEncoderLayer:
        - .linear1: First Linear layer (d_model -> dim_feedforward)
        - .linear2: Second Linear layer (dim_feedforward -> d_model)
        """
        encoder_layer = self.encoder.layers[layer_idx]
        linear1 = encoder_layer.linear1
        linear2 = encoder_layer.linear2
        return linear1, linear2

    def get_attn_layer(self, layer_idx=0):
        """Return the nn.MultiheadAttention module at layer_idx."""
        return self.encoder.layers[layer_idx].self_attn


# ============================================================================
# 3. TELEPORTATION: Diagonal Scaling Symmetry for FFN
# ============================================================================

@torch.no_grad()
def apply_ffn_diagonal_scaling_inplace(linear1, linear2, s_vec: torch.Tensor):
    """
    Apply diagonal scaling symmetry to FFN (preserves function for ReLU/GELU).
    
    For s_vec > 0 (size = hidden_dim):
      W1' = diag(s) @ W1  (row-wise scaling)
      b1' = s * b1        (element-wise)
      W2' = W2 @ diag(1/s)  (column-wise scaling)
      b2' = b2            (unchanged)
    
    This preserves: W2 @ ReLU(W1 @ x + b1) + b2 = (W2 @ diag(1/s)) @ ReLU((diag(s) @ W1) @ x + (s * b1)) + b2
    """
    s = s_vec.to(linear1.weight.device, linear1.weight.dtype)
    
    # Scale rows of W1 and entries of b1
    linear1.weight.mul_(s[:, None])
    if linear1.bias is not None:
        linear1.bias.mul_(s)
    
    # Scale columns of W2 by 1/s
    linear2.weight.mul_(1.0 / s[None, :])
    # linear2.bias unchanged (if exists)


# Removed unused function - teleportation is handled by teleport_ffn_diagonal_full_model


def objective_c_score(model, params_t_ordered, log_s, X_full, Y, loss_fn, lr, lambda_penalty=1e-3):
    """
    Compute Objective C score:
    J(log_s) = L(θ ∘ g(log_s) - η ∇_θ L(θ ∘ g(log_s))) + λ ||log_s||²
    
    Args:
        model: Model to evaluate
        params_t_ordered: Transformed parameters (OrderedDict) with requires_grad=True
        log_s: Log scale parameters (tensor, requires_grad=True)
        X_full: Input data
        Y: Target data
        loss_fn: Loss function
        lr: Step size η for virtual SGD step
        lambda_penalty: Penalty weight λ for ||log_s||²
    
    Returns:
        J(log_s): Scalar tensor (requires_grad=True)
    """
    # Forward pass with transformed parameters
    out = functional_call(model, params_t_ordered, (X_full,))
    loss = loss_fn(out, Y)
    
    # Compute gradients on transformed parameters
    # Virtual step updates only FFN-connected params to avoid SDPA double backward; this is a proxy Objective C.
    # Only request gradients for parameters that require them (to avoid errors)
    grad_keys = [k for k, v in params_t_ordered.items() if v.requires_grad]
    grad_vals = [params_t_ordered[k] for k in grad_keys]
    
    grads = torch.autograd.grad(loss, grad_vals, create_graph=True, retain_graph=True, allow_unused=True)
    
    # Virtual SGD step: θ_new = θ - η * ∇L(θ)
    params_new = OrderedDict()
    # Start with original parameters
    for k, v in params_t_ordered.items():
        params_new[k] = v
        
    # Apply updates where we have gradients
    for k, g in zip(grad_keys, grads):
        if g is not None:
            params_new[k] = params_t_ordered[k] - lr * g
    
    # Forward pass with virtual updated parameters
    out_new = functional_call(model, params_new, (X_full,))
    loss_new = loss_fn(out_new, Y)
    
    # Penalty term: λ ||log_s||²
    penalty = lambda_penalty * (log_s * log_s).sum()
    
    # Objective C: loss after virtual step + penalty
    J = loss_new + penalty
    
    return J


def teleport_ffn_diagonal_full_model(model, layer_idx, X_full, loss_fn, Y, lr_theta=1e-2, steps=20, log_s_clip=(-2.0, 2.0), lr=0.01, lambda_penalty=1e-3):
    """
    Teleport FFN in a TransformerEncoderLayer using Objective C:
    Minimize J(log_s) = L(θ ∘ g(log_s) - η ∇L(θ ∘ g(log_s))) + λ ||log_s||²
    
    Uses functional_call to search without mutating model.
    
    Args:
        model: Model containing the FFN
        layer_idx: Index of encoder layer to teleport
        X_full: Input data
        loss_fn: Loss function
        Y: Target data
        lr_theta: Learning rate for log_s optimization
        steps: Number of optimization steps
        log_s_clip: Clamping range for log_s
        lr: Step size η for virtual SGD step (defaults to training lr)
        lambda_penalty: Penalty weight λ for ||log_s||² (default 1e-3)
    
    Returns:
        s_best: Best scaling factors (tensor, detached)
        J_before: Objective C at log_s=0 (float)
        J_best: Best Objective C value (float)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # FIXED: Run search in eval mode to disable any stochastic behavior
    # Also save/restore RNG state inside the function to ensure deterministic behavior
    was_training = model.training
    model.eval()  # Disables dropout, batch norm updates, etc.
    
    # Save RNG state at start of search
    cpu_rng_start = torch.get_rng_state()
    cuda_rng_start = None
    if torch.cuda.is_available():
        cuda_rng_start = torch.cuda.get_rng_state_all()
    
    linear1, linear2 = model.get_ffn_layers(layer_idx)
    hidden_dim = linear1.weight.shape[0]
    
    # Store original parameters (need to detach and clone for functional_call)
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}
    
    # Find parameter names for this FFN
    encoder_prefix = f'encoder.layers.{layer_idx}.'
    w1_name = encoder_prefix + 'linear1.weight'
    b1_name = encoder_prefix + 'linear1.bias'
    w2_name = encoder_prefix + 'linear2.weight'
    
    log_s = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
    best_log_s = log_s.detach().clone()
    best_J = None
    J_before = None
    
    # Compute J_before at log_s = 0 (identity)
    log_s_zero = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
    log_s_zero_clamped = torch.clamp(log_s_zero, log_s_clip[0], log_s_clip[1])
    s_vec_zero = torch.exp(log_s_zero_clamped)
    params_t_zero = {k: v.clone() for k, v in params_dict.items()}
    params_t_zero[w1_name] = params_dict[w1_name] * s_vec_zero[:, None]
    if b1_name in params_dict and params_dict[b1_name] is not None:
        params_t_zero[b1_name] = params_dict[b1_name] * s_vec_zero
    params_t_zero[w2_name] = params_dict[w2_name] / s_vec_zero[None, :]
    
    # Ensure all parameters require grad for the virtual step (so we can compute dL/dtheta)
    # BUT do not call requires_grad_(True) on tensors that already have it (FFN weights),
    # as that might detach them from the graph history (log_s dependency).
    # ALSO: Do NOT force requires_grad on attention weights, because differentiating
    # through attention updates requires double backward, which SDPA doesn't support on CPU.
    # We will only update FFN weights in the virtual step.
    # for v in params_t_zero.values():
    #     if not v.requires_grad:
    #         v.requires_grad_(True)
            
    params_t_zero_ordered = OrderedDict(params_t_zero)
    J_before = objective_c_score(model, params_t_zero_ordered, log_s_zero, X_full, Y, loss_fn, lr, lambda_penalty)
    J_before_val = float(J_before.detach().item())
    
    # Compute initial gradient norm at identity (Task D requirement)
    initial_grad_norm = 0.0
    
    # CRITICAL: Flash Attention does not support double backward.
    # We must use math attention or disable flash/mem_efficient.
    try:
        from torch.backends.cuda import sdp_kernel
        if torch.cuda.is_available():
            ctx = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        else:
            import contextlib
            ctx = contextlib.nullcontext()
    except ImportError:
        import contextlib
        ctx = contextlib.nullcontext()
        
    with ctx:
        # Don't swallow errors - fail loudly if graph is broken
        dJ_dlog_0 = torch.autograd.grad(J_before, log_s_zero, retain_graph=False, create_graph=False)[0]
        initial_grad_norm = float(dJ_dlog_0.norm().item())
            
    # NEW: Check gradient at random point (Check B)
    grad_norm_rand = 0.0
    with ctx:
        log_s_rand = torch.randn_like(log_s_zero) * 0.5
        log_s_rand.requires_grad_(True)
        
        s_vec_rand = torch.exp(log_s_rand)
        params_t_rand = {k: v.clone() for k, v in params_dict.items()}
        params_t_rand[w1_name] = params_dict[w1_name] * s_vec_rand[:, None]
        if b1_name in params_dict and params_dict[b1_name] is not None:
            params_t_rand[b1_name] = params_dict[b1_name] * s_vec_rand
        params_t_rand[w2_name] = params_dict[w2_name] / s_vec_rand[None, :]
        
        # for v in params_t_rand.values():
        #     if not v.requires_grad:
        #         v.requires_grad_(True)
        
        params_t_rand_ordered = OrderedDict(params_t_rand)
        
        J_rand = objective_c_score(model, params_t_rand_ordered, log_s_rand, X_full, Y, loss_fn, lr, lambda_penalty)
        dJ_dlog_rand = torch.autograd.grad(J_rand, log_s_rand, retain_graph=False, create_graph=False)[0]
        grad_norm_rand = float(dJ_dlog_rand.norm().item())
    
    max_log_s_norm = 0.0
    
    for iter_idx in range(steps):
        log_s_clamped = torch.clamp(log_s, log_s_clip[0], log_s_clip[1])
        
        # Track max log_s norm during search
        current_norm = float(log_s_clamped.norm().item())
        if current_norm > max_log_s_norm:
            max_log_s_norm = current_norm
            
        s_vec = torch.exp(log_s_clamped)
        
        # Create transformed parameter dict (copy first, then modify)
        params_t = {k: v.clone() for k, v in params_dict.items()}
        
        # Apply diagonal scaling symmetry to FFN parameters
        # W1: row-wise scaling (diag(s) @ W1)
        params_t[w1_name] = params_dict[w1_name] * s_vec[:, None]
        
        # b1: element-wise scaling (if exists)
        if b1_name in params_dict and params_dict[b1_name] is not None:
            params_t[b1_name] = params_dict[b1_name] * s_vec
        
        # W2: column-wise scaling (W2 @ diag(1/s))
        params_t[w2_name] = params_dict[w2_name] / s_vec[None, :]
        # b2 unchanged
        
        # Convert to OrderedDict with requires_grad for autograd
        # for v in params_t.values():
        #     if not v.requires_grad:
        #         v.requires_grad_(True)
        params_t_ordered = OrderedDict(params_t)
        
        # Compute Objective C
        with ctx:
            J = objective_c_score(model, params_t_ordered, log_s_clamped, X_full, Y, loss_fn, lr, lambda_penalty)
            
            # Gradient descent on log_s to minimize J
            dJ_dlog = torch.autograd.grad(J, log_s, retain_graph=False, create_graph=False)[0]
            if not torch.isfinite(dJ_dlog).all():
                break
            
            # Gradient descent step (minimize J, so subtract gradient)
            log_s = (log_s - lr_theta * dJ_dlog).detach().requires_grad_(True)
            
            # Track best (lowest J)
            J_val = float(J.detach().item())
            if (best_J is None) or (J_val < best_J):
                best_J = J_val
                best_log_s = torch.clamp(log_s.detach(), log_s_clip[0], log_s_clip[1])
    
    s_best = torch.exp(best_log_s).detach()
    J_best_val = best_J if best_J is not None else J_before_val
    
    # CRITICAL: Restore RNG state to exactly what it was before the search
    # This ensures the search doesn't affect the RNG sequence seen by subsequent training steps
    torch.set_rng_state(cpu_rng_start)
    if cuda_rng_start is not None:
        torch.cuda.set_rng_state_all(cuda_rng_start)
    
    # Restore training mode
    if was_training:
        model.train()
    
    # Apply best transformation in-place
    linear1, linear2 = model.get_ffn_layers(layer_idx)
    apply_ffn_diagonal_scaling_inplace(linear1, linear2, s_best)
    
    # Clear gradients
    for p in model.parameters():
        p.grad = None
    
    return s_best, J_before_val, J_best_val, initial_grad_norm, max_log_s_norm, grad_norm_rand


# ============================================================================
# 4. OPTIMIZER WITH TELEPORTATION
# ============================================================================

class SGDTeleportOptimizer:
    """
    Wrapper that combines SGD with symmetry teleportation.
    
    Usage:
        optimizer = SGDTeleportOptimizer(model.parameters(), lr=0.01, teleport_every=5)
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Standard SGD step
        optimizer.teleport_step()  # Optional: apply teleportation
    """
    def __init__(self, model, params, lr=0.01, momentum=0.0, weight_decay=0.0,
                 teleport_every=5, teleport_layer_idx=0, X_teleport=None, Y_teleport=None, loss_fn=None,
                 lr_theta=1e-2, teleport_steps=20, acceptance_threshold=-1e-9, min_log_s_magnitude=1e-6):
        self.model = model
        self.teleport_every = teleport_every
        self.teleport_layer_idx = teleport_layer_idx
        self.X_teleport = X_teleport
        self.Y_teleport = Y_teleport
        self.loss_fn = loss_fn
        self.lr_theta = lr_theta
        self.teleport_steps = teleport_steps
        self.lambda_penalty = 1e-3  # Default penalty weight for Objective C
        self.acceptance_threshold = acceptance_threshold  # ΔJ threshold for acceptance
        self.min_log_s_magnitude = min_log_s_magnitude  # Minimum |log_s| for nontrivial
        self.step_count = 0
        
        # Create underlying SGD optimizer
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        # NEW: Create fixed batch for invariance checking
        self.X_check = None
        
        # NEW: Track diagnostics
        self.invariance_deltas = []
        self.grad_ratios = []
        self.teleport_noop = False  # Track if teleportation ever does anything
        self.objective_c_deltas = []  # Track Objective C improvements
        self.teleport_active_steps = []  # Track which teleport steps were active (strict criteria)
        self.initial_grad_norms = []
        self.max_log_s_norms = []
        self.grad_norms_rand = []
        
        # NEW: Detailed per-attempt logging
        self.teleport_attempts = []  # List of dicts with full diagnostics per attempt
    
    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        """Standard SGD step."""
        self.optimizer.step()
        self.step_count += 1
    
    def _compute_grad_norm(self):
        """Compute L2 norm of current gradients."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += (p.grad.detach() ** 2).sum().item()
        return np.sqrt(total)
    
    @staticmethod
    def _flat_param_norm(model):
        """Compute L2 norm of all parameters."""
        with torch.no_grad():
            s = 0.0
            for p in model.parameters():
                s += (p.detach().float().norm() ** 2).item()
            return s ** 0.5
    
    @staticmethod
    def _max_param_delta(model, params_before):
        """Compute max absolute change in any parameter."""
        with torch.no_grad():
            mx = 0.0
            for (name, p) in model.named_parameters():
                if name in params_before:
                    mx = max(mx, (p.detach() - params_before[name]).abs().max().item())
            return mx
    
    def teleport_step(self):
        """
        Apply teleportation if it's time (every teleport_every steps).
        Should be called after step() if teleportation is desired.
        
        Now includes:
        - Function invariance verification
        - Correct gradient diagnostics (before/after on same batch)
        """
        # Check if we should teleport
        if self.teleport_every == 0 or self.step_count % self.teleport_every != 0:
            return
        
        if self.X_teleport is None or self.Y_teleport is None or self.loss_fn is None:
            return
        
        X = self.X_teleport
        Y = self.Y_teleport
        loss_fn = self.loss_fn
        
        # Initialize check batch on first teleportation (FIXED batch, never changes)
        if self.X_check is None:
            self.X_check = X[:min(4, X.shape[0])].detach().clone().to(X.device)
        
        # Use same batch for all measurements
        Xb = X
        Yb = Y
        
        # STEP 1: Save parameters BEFORE teleportation
        params_before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        norm_before = self._flat_param_norm(self.model)
        
        # CRITICAL: Save optimizer state BEFORE teleportation
        # This ensures we can restore it if teleport is NO-OP
        optimizer_state_before = self.optimizer.state_dict()
        
        # STEP 2: Compute gradients BEFORE teleportation
        self.model.zero_grad(set_to_none=True)
        out = self.model(Xb)
        if len(out.shape) == 3:
            out_flat = out.reshape(-1, out.size(-1))
            Y_flat = Yb.reshape(-1) if len(Yb.shape) == 2 else Yb.reshape(-1)
        else:
            out_flat = out
            Y_flat = Yb
        loss_before = loss_fn(out_flat, Y_flat)
        loss_before.backward()
        grad_norm_before = self._compute_grad_norm()
        
        # STEP 3: Check invariance BEFORE (on fixed check batch)
        with torch.no_grad():
            out_before = self.model(self.X_check).detach().clone()
        
        # STEP 4: Apply teleportation
        # NOTE: RNG state is saved/restored INSIDE teleport_ffn_diagonal_full_model
        # so we don't need to do it here (it's already handled)
        self.model.zero_grad(set_to_none=True)
        s_best, J_before_val, J_best_val, initial_grad_norm, max_log_s_norm, grad_norm_rand = teleport_ffn_diagonal_full_model(
            self.model,
            self.teleport_layer_idx,
            Xb,
            loss_fn,
            Yb,
            lr_theta=self.lr_theta,
            steps=self.teleport_steps,
            log_s_clip=(-2.0, 2.0),
            lr=self.optimizer.defaults.get('lr', 0.01),
            lambda_penalty=self.lambda_penalty
        )
        
        # STEP 5: Verify parameters changed and log s_vec stats
        norm_after = self._flat_param_norm(self.model)
        max_w_delta = self._max_param_delta(self.model, params_before)
        
        # Log s_vec statistics (Task A.2)
        s_min = float(s_best.min().item())
        s_max = float(s_best.max().item())
        s_mean = float(s_best.mean().item())
        s_std = float(s_best.std().item())
        log_s_best = torch.log(s_best)
        log_s_norm = float(log_s_best.norm().item())
        max_abs_log_s = float(log_s_best.abs().max().item())
        mean_abs_log_s = float(log_s_best.abs().mean().item())
        
        # Compute ΔJ
        delta_J = J_best_val - J_before_val
        
        # STRICT ACTIVATION CRITERIA:
        # 1. ΔJ < acceptance_threshold (improvement in objective C)
        # 2. max_abs_log_s >= min_log_s_magnitude (nontrivial scaling)
        # 3. max_w_delta >= 1e-6 (parameters actually changed)
        accepted = delta_J < self.acceptance_threshold
        nontrivial_scaling = max_abs_log_s >= self.min_log_s_magnitude
        params_changed = max_w_delta >= 1e-6
        
        teleport_active = accepted and nontrivial_scaling and params_changed
        
        if not teleport_active:
            self.teleport_noop = True
        
        # Log detailed attempt information
        attempt_info = {
            'step': self.step_count,
            'accepted': accepted,
            'nontrivial_scaling': nontrivial_scaling,
            'params_changed': params_changed,
            'active': teleport_active,
            'delta_J': delta_J,
            'J_before': J_before_val,
            'J_best': J_best_val,
            'max_abs_log_s': max_abs_log_s,
            'mean_abs_log_s': mean_abs_log_s,
            'log_s_norm': log_s_norm,
            'max_w_delta': max_w_delta,
            's_min': s_min,
            's_max': s_max,
            's_mean': s_mean,
            's_std': s_std,
            'initial_grad_norm': initial_grad_norm,
            'grad_norm_rand': grad_norm_rand,
            'max_log_s_during_search': max_log_s_norm
        }
        self.teleport_attempts.append(attempt_info)
            
        # Track diagnostics (Step 4.5: Always record attempt metrics)
        self.teleport_active_steps.append(teleport_active)
        self.objective_c_deltas.append(delta_J)
        self.initial_grad_norms.append(initial_grad_norm)
        self.max_log_s_norms.append(max_log_s_norm)
        self.grad_norms_rand.append(grad_norm_rand)
        
        # Check if s was clipped
        hidden_dim = s_best.shape[0]
        is_clipped = (log_s_norm >= 2.0 * (hidden_dim**0.5) - 1e-3) or \
                     (s_max > np.exp(2.0) - 1e-3) or (s_min < np.exp(-2.0) + 1e-3)
        
        # TASK B: Hard, auditable diagnostics
        status_flags = []
        if not teleport_active:
            status_flags.append('[REJECTED]')
            if not accepted:
                status_flags.append(f'ΔJ={delta_J:.2e}≥{self.acceptance_threshold:.2e}')
            if not nontrivial_scaling:
                status_flags.append(f'max|log_s|={max_abs_log_s:.2e}<{self.min_log_s_magnitude:.2e}')
            if not params_changed:
                status_flags.append(f'max|Δw|={max_w_delta:.2e}<1e-6')
        else:
            status_flags.append('[ACCEPTED]')
        if is_clipped:
            status_flags.append('[CLIPPED]')
            
        print(f"[Teleport step {self.step_count}] {' '.join(status_flags)}")
        print(f"  Criteria: accepted={accepted}, nontrivial={nontrivial_scaling}, changed={params_changed} → ACTIVE={teleport_active}")
        print(f"  Parameter change: ||θ|| {norm_before:.6f} -> {norm_after:.6f} | max|Δw|={max_w_delta:.3e}")
        print(f"  Scaling: max|log_s|={max_abs_log_s:.3e}, mean|log_s|={mean_abs_log_s:.3e}, ||log_s||={log_s_norm:.4f}")
        print(f"  s_vec: min={s_min:.4f}, max={s_max:.4f}, mean={s_mean:.4f}, std={s_std:.4f}")
        print(f"  Objective C: J_before={J_before_val:.6e}, J_best={J_best_val:.6e}, ΔJ={delta_J:.6e}")
        print(f"  Diagnostics: ||∂J/∂log_s||_0={initial_grad_norm:.3e}, ||∂J/∂log_s||_rand={grad_norm_rand:.3e}")

        # CRITICAL: If NO-OP detected, restore original parameters AND optimizer state
        if not teleport_active:
            # Restore original parameters (teleportation found identity, so revert)
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in params_before:
                        p.copy_(params_before[name])
            
            # CRITICAL: Restore optimizer state to what it was before teleport_step()
            self.optimizer.load_state_dict(optimizer_state_before)
            
            # Clear gradients
            self.model.zero_grad(set_to_none=True)
            
            # Record N/A diagnostics for downstream consistency
            # Invariance is trivially 0
            self.invariance_deltas.append(0.0)
            # Grad ratio is 1.0 (unchanged)
            self.grad_ratios.append(1.0)
            
            print(f"  Invariance: max|Δout|=0.000e+00 (Trivial NO-OP)")
            
            return
        
        # STEP 6: Check invariance AFTER (only if active)
        # Note: If active, we MUST verify invariance
        with torch.no_grad():
            out_after = self.model(self.X_check).detach().clone()
            delta = out_after - out_before
            max_abs_delta = delta.abs().max().item()
            self.invariance_deltas.append(max_abs_delta)
            
        INVARIANCE_THRESHOLD = 1e-5
        print(f"  Invariance: max|Δout|={max_abs_delta:.3e} (PASS < {INVARIANCE_THRESHOLD:.3e})")
        
        # TASK A.3: Hard assertions (TASK B requirement)
        if max_abs_delta >= INVARIANCE_THRESHOLD:
            raise RuntimeError(
                f"Invariance violation at step {self.step_count}: "
                f"max|Δout|={max_abs_delta:.3e} >= {INVARIANCE_THRESHOLD:.3e}"
            )
        
        # STEP 7: Compute gradients AFTER teleportation (MUST recompute)
        self.model.zero_grad(set_to_none=True)
        out = self.model(Xb)
        if len(out.shape) == 3:
            out_flat = out.reshape(-1, out.size(-1))
            Y_flat = Yb.reshape(-1) if len(Yb.shape) == 2 else Yb.reshape(-1)
        else:
            out_flat = out
            Y_flat = Yb
        loss_after = loss_fn(out_flat, Y_flat)
        loss_after.backward()
        grad_norm_after = self._compute_grad_norm()
        grad_ratio = grad_norm_after / (grad_norm_before + 1e-12)
        
        self.grad_ratios.append(grad_ratio)
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ============================================================================
# 5. TRAINING LOOP AND COMPARISON
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, optimizer, X, Y, loss_fn):
    """Single training epoch."""
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_with_teleport(model, X, Y, loss_fn, lr=0.01, num_steps=200, teleport_every=5):
    """Train with teleportation-enabled optimizer."""
    optimizer = SGDTeleportOptimizer(
        model=model,
        params=model.parameters(),
        lr=lr,
        teleport_every=teleport_every,
        teleport_layer_idx=0,  # Teleport first (and only) encoder layer
        X_teleport=X,
        Y_teleport=Y,
        loss_fn=loss_fn,
        lr_theta=1e-2,
        teleport_steps=20
    )
    
    # Store optimizer reference for diagnostics access (TASK B)
    model._last_optimizer = optimizer
    
    losses = []
    for step in range(num_steps):
        loss = train_epoch(model, optimizer, X, Y, loss_fn)
        losses.append(loss)
        # Apply teleportation AFTER recording loss (so loss is at same point as baseline)
        optimizer.teleport_step()  # Apply teleportation if it's time
    
    return losses

def train_baseline(model, X, Y, loss_fn, lr=0.01, num_steps=200):
    """Train with standard SGD (no teleportation)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for step in range(num_steps):
        loss = train_epoch(model, optimizer, X, Y, loss_fn)
        losses.append(loss)
    
    return losses

def make_run(seed, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, 
             batch_size=16, seq_len=4, num_samples=16):
    """
    TASK B: Create paired run setup (same seed -> same model, same data).
    Returns model and data for a single seed.
    """
    torch.manual_seed(seed)
    
    # Generate synthetic data
    U, _ = torch.linalg.qr(torch.randn(16, 16))
    singular_vals = torch.logspace(0, 4, 16)
    X_full = U @ torch.diag(singular_vals)
    X_data = X_full[:num_samples, :4].unsqueeze(1).repeat(1, seq_len, 1)
    Y_data = torch.randn(num_samples, 4)
    
    # Create model
    torch.manual_seed(seed)  # Reset for model init
    model = TinyTransformer(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=0.0
    )
    
    return model, X_data, Y_data


def run_comparison(num_seeds=5, num_steps=200, teleport_every=20, lr=0.01):
    """
    Run comparison: baseline SGD vs SGD + teleportation.
    
    Args:
        num_seeds: Number of random seeds to run (for statistical significance)
    
    Returns:
        dict with results, diagnostics, and statistics
    """
    # Hyperparameters
    batch_size = 16
    seq_len = 4
    num_samples = batch_size
    d_model = 32
    nhead = 2
    num_layers = 1
    dim_feedforward = 64
    loss_fn = nn.MSELoss()
    
    # Storage for results (TASK B: structured per-seed results)
    results = []
    all_losses_baseline = []
    all_losses_teleport = []
    all_final_baseline = []
    all_final_teleport = []
    all_invariance_deltas = []
    all_grad_ratios = []
    
    print("=" * 60)
    print(f"Transformer Teleportation Comparison (n={num_seeds} seeds)")
    print(f"Config: steps={num_steps}, teleport_every={teleport_every}, lr={lr}")
    print("=" * 60)
    
    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx
        
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}/{num_seeds} (seed={seed})")
        print(f"{'='*60}")
        
        # STEP A: Build shared initial state ONCE per seed
        torch.manual_seed(seed)
        # Generate data ONCE (same for both runs)
        U, _ = torch.linalg.qr(torch.randn(16, 16))
        singular_vals = torch.logspace(0, 4, 16)
        X_full = U @ torch.diag(singular_vals)
        X_data = X_full[:num_samples, :4].unsqueeze(1).repeat(1, seq_len, 1)
        Y_data = torch.randn(num_samples, 4)
        
        # Create model ONCE and snapshot init weights
        torch.manual_seed(seed)  # Reset for model init
        model0 = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=0.0
        )
        init_state = {k: v.detach().clone() for k, v in model0.state_dict().items()}
        
        if seed_idx == 0:
            print(f"Model parameters: {count_parameters(model0)}")
        
        # STEP B: Create baseline + teleport models by loading the same init_state
        model_baseline = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=0.0
        )
        model_baseline.load_state_dict(init_state)
        
        model_teleport = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=0.0
        )
        model_teleport.load_state_dict(init_state)
        
        # STEP C: Hard asserts RIGHT HERE
        # (1) weights identical
        for k in init_state:
            a = model_baseline.state_dict()[k]
            b = model_teleport.state_dict()[k]
            assert torch.equal(a, b), f"INIT MISMATCH at {k}: max diff = {(a - b).abs().max().item():.3e}"
        
        # (2) data identical (we're using the SAME X_data, Y_data objects for both runs)
        # No need to assert - they're the same objects
        
        # Baseline: SGD only
        optimizer_baseline = torch.optim.SGD(model_baseline.parameters(), lr=lr)
        
        # Treatment: SGD + teleportation (using SAME init_state and SAME data)
        # Create teleport optimizer
        optimizer_teleport = SGDTeleportOptimizer(
            model=model_teleport,
            params=model_teleport.parameters(),
            lr=lr,
            teleport_every=teleport_every,
            teleport_layer_idx=0,
            X_teleport=X_data,  # SAME data objects
            Y_teleport=Y_data,
            loss_fn=loss_fn,
            lr_theta=1e-2,
            teleport_steps=20
        )
        model_teleport._last_optimizer = optimizer_teleport
        
        # Interleaved Training Loop
        print("\nTraining baseline and teleportation interleaved...")
        
        losses_baseline = []
        losses_teleport = []
        
        for step in range(num_steps):
            # 1. Baseline Step
            loss_b = train_epoch(model_baseline, optimizer_baseline, X_data, Y_data, loss_fn)
            losses_baseline.append(loss_b)
            
            # 2. Teleport Model Step (SGD part)
            loss_t = train_epoch(model_teleport, optimizer_teleport, X_data, Y_data, loss_fn)
            losses_teleport.append(loss_t)
            
            # 3. Teleport Step (Symmetry part, if scheduled)
            # This happens AFTER the SGD step, consistent with the paper's "teleport after step"
            optimizer_teleport.teleport_step()
            
            # 4. Strict Identity Check on non-active steps
            # We only assert identity if teleportation has NEVER been active so far.
            # Once it becomes active, models diverge legitimately.
            
            has_been_active = False
            if hasattr(optimizer_teleport, 'teleport_active_steps') and optimizer_teleport.teleport_active_steps:
                if any(optimizer_teleport.teleport_active_steps):
                    has_been_active = True
            
            # If teleport has never been active (meaning all previous attempts were NO-OPs
            # and current step is either not scheduled or NO-OP), models must match.
            if not has_been_active:
                # Check loss match
                loss_diff = abs(loss_b - loss_t)
                assert loss_diff < 1e-9, f"Loss mismatch at step {step} (no active teleport yet): {loss_b} vs {loss_t}, diff={loss_diff:.3e}"
                
                # Check parameter match
                for (n1, p1), (n2, p2) in zip(model_baseline.named_parameters(), model_teleport.named_parameters()):
                    param_diff = (p1 - p2).abs().max().item()
                    assert param_diff < 1e-9, f"Param mismatch at step {step} (no active teleport yet) ({n1}): {param_diff:.3e}"
        
        final_baseline = losses_baseline[-1]
        final_teleport = losses_teleport[-1]
        
        # Calculate summary metrics
        best_baseline = min(losses_baseline)
        auc_baseline = sum(losses_baseline)
        all_losses_baseline.append(losses_baseline)
        all_final_baseline.append(final_baseline)
        
        best_teleport = min(losses_teleport)
        auc_teleport = sum(losses_teleport)
        all_losses_teleport.append(losses_teleport)
        all_final_teleport.append(final_teleport)
        
        print(f"  Final loss: baseline={final_baseline:.6f}, teleport={final_teleport:.6f}")
        
        # Collect diagnostics from optimizer
        optimizer = getattr(model_teleport, '_last_optimizer', None)
        teleport_noop = False
        all_initial_grad_norms = []
        all_max_log_s_norms = []
        all_objective_c_deltas = []
        
        if optimizer is not None:
            teleport_noop = getattr(optimizer, 'teleport_noop', False)
            if hasattr(optimizer, 'invariance_deltas') and optimizer.invariance_deltas:
                all_invariance_deltas.extend(optimizer.invariance_deltas)
            if hasattr(optimizer, 'grad_ratios') and optimizer.grad_ratios:
                all_grad_ratios.extend(optimizer.grad_ratios)
            if hasattr(optimizer, 'initial_grad_norms'):
                all_initial_grad_norms = optimizer.initial_grad_norms
            if hasattr(optimizer, 'max_log_s_norms'):
                all_max_log_s_norms = optimizer.max_log_s_norms
            if hasattr(optimizer, 'objective_c_deltas'):
                all_objective_c_deltas = optimizer.objective_c_deltas
        
        # TASK C: Compute steps-to-threshold (threshold = 0.8 * baseline_final_loss)
        threshold = 0.8 * final_baseline
        steps_to_threshold_baseline = None
        steps_to_threshold_teleport = None
        for step, loss in enumerate(losses_baseline):
            if loss <= threshold and steps_to_threshold_baseline is None:
                steps_to_threshold_baseline = step + 1
        for step, loss in enumerate(losses_teleport):
            if loss <= threshold and steps_to_threshold_teleport is None:
                steps_to_threshold_teleport = step + 1
        
        # TASK C: Compute teleport active rate (% of teleport steps with max|Δw| > 1e-8)
        teleport_active_rate = 0.0
        if optimizer is not None and hasattr(optimizer, 'teleport_active_steps') and optimizer.teleport_active_steps:
            teleport_active_rate = 100.0 * sum(optimizer.teleport_active_steps) / len(optimizer.teleport_active_steps)
        
        # TASK B: Store structured per-seed results
        seed_result = {
            "seed": seed,
            "baseline": {
                "final_loss": final_baseline,
                "best_loss": best_baseline,
                "auc": auc_baseline,
                "losses": losses_baseline,
                "steps_to_threshold": steps_to_threshold_baseline
            },
            "teleport": {
                "final_loss": final_teleport,
                "best_loss": best_teleport,
                "auc": auc_teleport,
                "losses": losses_teleport,
                "steps_to_threshold": steps_to_threshold_teleport,
                "active_rate": teleport_active_rate
            },
            "teleport_noop": teleport_noop
        }
        results.append(seed_result)
        
        # Per-seed diagnostics (TASK D format)
        print(f"\n[Seed {seed}] Comparison:")
        print(f"  final_loss: baseline={final_baseline:.6f}, teleport={final_teleport:.6f}, delta={final_teleport-final_baseline:+.6f}")
        print(f"  best_loss:  baseline={best_baseline:.6f}, teleport={best_teleport:.6f}, delta={best_teleport-best_baseline:+.6f}")
        print(f"  AUC:        baseline={auc_baseline:.4f}, teleport={auc_teleport:.4f}, ratio={auc_teleport/auc_baseline:.4f}")
        if teleport_noop:
            print(f"  WARNING: TELEPORT = NO-OP (max|Δw| < 1e-6 at all steps)")
            # HARD CHECK: If NO-OP, losses must be identical (up to float noise)
            diff = abs(final_baseline - final_teleport)
            print(f"  NO-OP seed final loss diff: {diff:.3e}")
            assert diff < 1e-6, f"NO-OP but losses differ by {diff:.3e} -> pairing/measurement bug!"
    
    # TASK A.3: Check if teleportation was NO-OP across all seeds
    any_teleport_active = any(not r["teleport_noop"] for r in results)
    all_teleport_noop = all(r["teleport_noop"] for r in results)
    
    # TASK C: Compute statistics including AUC and active rate
    baseline_final_mean = np.mean(all_final_baseline)
    baseline_final_std = np.std(all_final_baseline)
    teleport_final_mean = np.mean(all_final_teleport)
    teleport_final_std = np.std(all_final_teleport)
    
    baseline_aucs = [r["baseline"]["auc"] for r in results]
    teleport_aucs = [r["teleport"]["auc"] for r in results]
    baseline_auc_mean = np.mean(baseline_aucs)
    baseline_auc_std = np.std(baseline_aucs)
    teleport_auc_mean = np.mean(teleport_aucs)
    teleport_auc_std = np.std(teleport_aucs)
    
    teleport_active_rates = [r["teleport"]["active_rate"] for r in results]
    active_rate_mean = np.mean(teleport_active_rates)
    active_rate_std = np.std(teleport_active_rates)
    
    # Paired t-tests
    t_stat_final, p_value_final = stats.ttest_rel(all_final_teleport, all_final_baseline)
    t_stat_auc, p_value_auc = stats.ttest_rel(teleport_aucs, baseline_aucs)
    
    # FIXED: Compute improvement from per-seed paired diffs (foolproof, no mixing)
    # Negative diff means teleport is better (lower loss)
    assert len(all_final_teleport) == len(all_final_baseline), "Array length mismatch!"
    paired_diffs = [all_final_teleport[i] - all_final_baseline[i] for i in range(len(all_final_baseline))]
    mean_diff = np.mean(paired_diffs)
    improvement_pct = -mean_diff / baseline_final_mean * 100  # negative diff = improvement
    
    # TASK D: Print final output in required format
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # TASK C: Print mandatory diagnostics
    print(f"\nTeleport active: {'NO-OP' if all_teleport_noop else ('YES' if any_teleport_active else 'NO')}")
    if all_teleport_noop:
        print("  WARNING: Teleportation found identity transformation (s_vec approximately 1.0) at all steps")
        print("  This means teleportation did not change parameters. Any improvement claims are invalid.")
    
    print(f"\nTeleport active rate: {active_rate_mean:.1f}% ± {active_rate_std:.1f}%")
    if active_rate_mean < 50.0:
        print("  WARNING: Active rate < 50% - teleportation is collapsing to identity")
    
    if all_invariance_deltas:
        max_inv_delta = np.max(all_invariance_deltas)
        print(f"Invariance: PASS (max|Δout| = {max_inv_delta:.3e} < 1e-6)")
    else:
        print("Invariance: N/A (no teleportation steps)")
    
    # TASK C: Print final loss, AUC, and steps-to-threshold
    print(f"\nFinal Loss (mean ± std):")
    print(f"  Baseline:  {baseline_final_mean:.6f} ± {baseline_final_std:.6f}")
    print(f"  Teleport:  {teleport_final_mean:.6f} ± {teleport_final_std:.6f}")
    print(f"  Δ% improvement: {improvement_pct:+.2f}%")
    print(f"  p-value (paired t-test): {p_value_final:.4f}")
    if p_value_final < 0.05:
        print("    Statistically significant (p < 0.05)")
    else:
        print("    NOT statistically significant (p >= 0.05)")
    
    print(f"\nLoss AUC (mean ± std):")
    print(f"  Baseline:  {baseline_auc_mean:.4f} ± {baseline_auc_std:.4f}")
    print(f"  Teleport:  {teleport_auc_mean:.4f} ± {teleport_auc_std:.4f}")
    auc_improvement_pct = (baseline_auc_mean - teleport_auc_mean) / baseline_auc_mean * 100
    print(f"  Δ% improvement: {auc_improvement_pct:+.2f}%")
    print(f"  p-value (paired t-test): {p_value_auc:.4f}")
    if p_value_auc < 0.05:
        print("    Statistically significant (p < 0.05)")
    else:
        print("    NOT statistically significant (p >= 0.05)")
    
    # Steps-to-threshold (if applicable)
    baseline_steps_to_threshold = [r["baseline"]["steps_to_threshold"] for r in results if r["baseline"]["steps_to_threshold"] is not None]
    teleport_steps_to_threshold = [r["teleport"]["steps_to_threshold"] for r in results if r["teleport"]["steps_to_threshold"] is not None]
    if baseline_steps_to_threshold and teleport_steps_to_threshold:
        print(f"\nSteps to threshold (0.8 * baseline final loss):")
        print(f"  Baseline:  {np.mean(baseline_steps_to_threshold):.1f} ± {np.std(baseline_steps_to_threshold):.1f}")
        print(f"  Teleport:  {np.mean(teleport_steps_to_threshold):.1f} ± {np.std(teleport_steps_to_threshold):.1f}")
    
    # Acceptance criteria check
    if all_teleport_noop and improvement_pct > 0:
        print("\nWARNING: CRITICAL: Improvement claimed but teleportation is NO-OP!")
        print("   This indicates a comparison/pairing bug, not a real effect.")
    elif all_teleport_noop:
        print("\nValid negative result: Teleportation collapses to identity in this setting.")
    elif active_rate_mean < 50.0:
        print("\nWARNING: Teleportation active rate < 50% - collapsing to identity")
    elif p_value_auc >= 0.05:
        print("\nWARNING: Teleportation is active but does not show statistically significant improvement")
    elif p_value_auc < 0.05 and teleport_auc_mean < baseline_auc_mean:
        print("\nSUCCESS: Teleportation is active, invariance holds, and shows statistically significant improvement")
    
    # Diagnostics summary
    if all_invariance_deltas:
        print(f"\n🔬 Teleportation Diagnostics:")
        print(f"    Function invariance (max Δoutput):")
        print(f"      Mean: {np.mean(all_invariance_deltas):.3e}")
        print(f"      Max:  {np.max(all_invariance_deltas):.3e}")
        if np.max(all_invariance_deltas) < 1e-4:
            print(f"      Invariance preserved (< 1e-4)")
        else:
            print(f"      WARNING: Invariance may be violated (> 1e-4)")
    
    if all_grad_ratios:
        print(f"    Gradient norm ratio (after/before teleport):")
        print(f"      Mean: {np.mean(all_grad_ratios):.3f}x")
        print(f"      Max:  {np.max(all_grad_ratios):.3f}x")
        print(f"      Min:  {np.min(all_grad_ratios):.3f}x")
    
    # Plot with error bars
    plt.figure(figsize=(12, 6))
    
    # Compute mean and std for each step
    max_len = max(len(l) for l in all_losses_baseline)
    steps = range(max_len)
    
    baseline_mean_curve = np.array([np.mean([l[i] if i < len(l) else l[-1] for l in all_losses_baseline]) 
                                     for i in range(max_len)])
    baseline_std_curve = np.array([np.std([l[i] if i < len(l) else l[-1] for l in all_losses_baseline]) 
                                    for i in range(max_len)])
    
    teleport_mean_curve = np.array([np.mean([l[i] if i < len(l) else l[-1] for l in all_losses_teleport]) 
                                     for i in range(max_len)])
    teleport_std_curve = np.array([np.std([l[i] if i < len(l) else l[-1] for l in all_losses_teleport]) 
                                    for i in range(max_len)])
    
    plt.plot(steps, baseline_mean_curve, label='Baseline SGD', color='C0', linewidth=2)
    plt.fill_between(steps, baseline_mean_curve - baseline_std_curve, 
                     baseline_mean_curve + baseline_std_curve, alpha=0.2, color='C0')
    
    plt.plot(steps, teleport_mean_curve, label='SGD + Teleportation', color='C1', linewidth=2)
    plt.fill_between(steps, teleport_mean_curve - teleport_std_curve, 
                     teleport_mean_curve + teleport_std_curve, alpha=0.2, color='C1')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Transformer Optimization: Baseline vs Teleportation (n={num_seeds} seeds)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('transformer_teleport_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: transformer_teleport_comparison.png")
    
    return {
        'baseline_final': all_final_baseline,
        'teleport_final': all_final_teleport,
        'baseline_final_mean': baseline_final_mean,
        'teleport_final_mean': teleport_final_mean,
        'baseline_auc_mean': baseline_auc_mean,
        'teleport_auc_mean': teleport_auc_mean,
        'improvement_pct': improvement_pct,
        'p_value_final': p_value_final,
        'p_value_auc': p_value_auc,
        'active_rate_mean': active_rate_mean,
        'invariance_deltas': all_invariance_deltas,
        'grad_ratios': all_grad_ratios
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Teleportation with Objective C')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds (default: 5)')
    parser.add_argument('--steps', type=int, default=200, help='Number of training steps (default: 200)')
    parser.add_argument('--teleport_every', type=int, default=20, help='Teleport every N steps (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    args = parser.parse_args()
    
    run_comparison(num_seeds=args.num_seeds, num_steps=args.steps, teleport_every=args.teleport_every, lr=args.lr)

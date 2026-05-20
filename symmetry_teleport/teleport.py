"""
Core teleportation logic for symmetry-based optimization.

This module implements the Objective C minimization and teleportation search.
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict
import contextlib
from torch.nn.attention import sdpa_kernel, SDPBackend


def _math_sdpa():
    """Context manager that forces the math SDPA kernel.

    Used inside teleport_qk_diagonal: when in_proj_weight is built with
    requires_grad=True (through sym_param), flash-attention-for-cpu cannot
    propagate create_graph=True gradients.  The math kernel handles it fine.
    """
    return sdpa_kernel(SDPBackend.MATH)


def sym_param_to_s_vec(sym_param, s_param='exp', log_s_clip=(-2.0, 2.0), s_eps=1e-8):
    """
    Convert symmetry parameter to scaling vector s.
    
    Args:
        sym_param: Symmetry parameter tensor
        s_param: 'exp', 'direct', or 'projected'
        log_s_clip: Clamping range for log_s (only used for 'exp')
        s_eps: Small positive constant for numerical stability
    
    Returns:
        s_vec: Scaling vector (all positive)
    """
    if s_param == 'exp':
        # log_s parameterization: s = exp(clamp(log_s))
        log_s_clamped = torch.clamp(sym_param, log_s_clip[0], log_s_clip[1])
        s_vec = torch.exp(log_s_clamped)
    elif s_param == 'direct':
        # Direct parameterization: s = softplus(u) + eps
        s_vec = F.softplus(sym_param) + s_eps
    elif s_param == 'projected':
        # Projected parameterization: s directly, already clamped to [eps, inf)
        s_vec = sym_param
    else:
        raise ValueError(f"Unknown s_param: {s_param}")
    
    return s_vec


def _grad_norm_from_grads(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += (g * g).sum()
    if isinstance(total, torch.Tensor):
        return float(total.sqrt().item())
    return 0.0


def _grad_norm_from_grad_map(grad_map, names):
    total = 0.0
    for name in names:
        g = grad_map.get(name)
        if g is not None:
            total += (g * g).sum()
    if isinstance(total, torch.Tensor):
        return float(total.sqrt().item())
    return 0.0


def objective_virtual_loss(model, params_t_ordered, sym_param, X_full, Y, loss_fn, lr, lambda_penalty=1e-3):
    """
    Compute virtual loss objective (Objective C):
    J(θ) = L(θ ∘ g(s) - η ∇_θ L(θ ∘ g(s))) + λ ||log(s)||²
    
    This is the default objective: loss after a virtual SGD step + regularization.
    
    Args:
        model: Model to evaluate
        params_t_ordered: Transformed parameters (OrderedDict) with requires_grad=True
        sym_param: Symmetry parameters (tensor, requires_grad=True)
                   Can be log_s or s depending on parameterization
        X_full: Input data
        Y: Target data
        loss_fn: Loss function
        lr: Step size η for virtual SGD step
        lambda_penalty: Penalty weight λ for regularization
    
    Returns:
        J: Scalar tensor (requires_grad=True)
    """
    # Forward pass with transformed parameters
    out = functional_call(model, params_t_ordered, (X_full,))
    loss = loss_fn(out, Y)
    
    # Compute gradients on transformed parameters
    # Only include parameters that have grad_fn (connected to computation graph)
    grad_keys = [k for k, v in params_t_ordered.items() if v.requires_grad and v.grad_fn is not None]
    grad_vals = [params_t_ordered[k] for k in grad_keys]
    
    if not grad_vals:
        # No parameters to compute gradients for - skip virtual step
        # Still need to maintain gradient connection through penalty
        penalty = lambda_penalty * (sym_param * sym_param).sum()
        # Use loss (which has grad_fn) + penalty (which depends on sym_param)
        J = loss + penalty
        return J
    
    grads = torch.autograd.grad(loss, grad_vals, create_graph=True, retain_graph=True, allow_unused=True)
    
    # Virtual SGD step: θ_new = θ - η * ∇L(θ)
    params_new = OrderedDict()
    for k, v in params_t_ordered.items():
        params_new[k] = v
        
    for k, g in zip(grad_keys, grads):
        if g is not None:
            params_new[k] = params_t_ordered[k] - lr * g
    
    # Forward pass with virtual updated parameters
    out_new = functional_call(model, params_new, (X_full,))
    loss_new = loss_fn(out_new, Y)
    
    # Penalty term: λ ||sym_param||²
    penalty = lambda_penalty * (sym_param * sym_param).sum()
    
    # Objective: loss after virtual step + penalty
    J = loss_new + penalty
    
    return J


def objective_grad_norm(model, params_t_ordered, sym_param, X_full, Y, loss_fn, lambda_penalty=1e-3):
    """
    Compute gradient norm objective (Bo's specification):
    J(θ) = -||∇_φ L(φ)||² + λ ||θ||²
    
    Where φ = transformed parameters, θ = symmetry parameters.
    No virtual SGD step - just maximize gradient norm on transformed parameters.
    
    Args:
        model: Model to evaluate
        params_t_ordered: Transformed parameters (OrderedDict) with requires_grad=True
        sym_param: Symmetry parameters (tensor, requires_grad=True)
        X_full: Input data
        Y: Target data
        loss_fn: Loss function
        lambda_penalty: Penalty weight λ for regularization on symmetry params
    
    Returns:
        J: Scalar tensor (requires_grad=True)
    """
    # Forward pass with transformed parameters
    out = functional_call(model, params_t_ordered, (X_full,))
    loss = loss_fn(out, Y)
    
    # Compute gradients on transformed parameters
    grad_keys = [k for k, v in params_t_ordered.items() if v.requires_grad and v.grad_fn is not None]
    grad_vals = [params_t_ordered[k] for k in grad_keys]
    
    if not grad_vals:
        # No gradients - return just penalty
        penalty = lambda_penalty * (sym_param * sym_param).sum()
        return penalty
    
    grads = torch.autograd.grad(loss, grad_vals, create_graph=True, retain_graph=True, allow_unused=True)
    
    # Compute squared gradient norm: sum_i ||g_i||^2
    grad_norm_sq = sum((g * g).sum() for g in grads if g is not None)
    
    # Penalty term on symmetry parameters
    penalty = lambda_penalty * (sym_param * sym_param).sum()
    
    # Objective: minimize J = -(grad_norm_sq) + lambda * ||theta||^2
    # (minimizing J maximizes grad_norm_sq)
    J = -grad_norm_sq + penalty
    
    return J


def objective_virtual_sgd_improve(
    model,
    params_t_ordered,
    sym_param,
    X_full,
    Y,
    loss_fn,
    lr,
    virtual_steps=5,
    lr_virtual_mult=2.0,
):
    """
    Optimize the true acceptance target for virtual_sgd_improve:
    J = L_tp_virtual = loss after K virtual SGD steps from transformed params.
    """
    params_cur = OrderedDict((k, v) for k, v in params_t_ordered.items())
    effective_lr = lr_virtual_mult * lr
    loss_cur = None

    for _ in range(virtual_steps):
        out = functional_call(model, params_cur, (X_full,))
        loss_cur = loss_fn(out, Y)

        grad_keys = [k for k, v in params_cur.items() if v.requires_grad and v.grad_fn is not None]
        grad_vals = [params_cur[k] for k in grad_keys]
        if not grad_vals:
            return loss_cur + 0.0 * sym_param.sum()

        grads = torch.autograd.grad(loss_cur, grad_vals, create_graph=True, retain_graph=True, allow_unused=True)
        params_next = OrderedDict((k, v) for k, v in params_cur.items())
        for k, g in zip(grad_keys, grads):
            if g is not None:
                params_next[k] = params_cur[k] - effective_lr * g
        params_cur = params_next

    out_final = functional_call(model, params_cur, (X_full,))
    loss_final = loss_fn(out_final, Y)
    return loss_final


def teleport_ffn_diagonal(model, layer_idx, X_full, loss_fn, Y, 
                          lr_theta=1e-2, steps=20, log_s_clip=(-2.0, 2.0), 
                          lr=0.01, lambda_penalty=1e-3,
                          objective='virtual_loss', s_param='exp',
                          force_nontrivial_s=False, theta_target_max_log_s=0.5,
                          debug_inner_log=False):
    """
    Teleport FFN using diagonal scaling symmetry.
    
    Supports objectives:
    - 'virtual_loss': J(θ) = L(θ ∘ g(s) - η ∇L(θ ∘ g(s))) + λ ||log(s)||²
    - 'grad_norm': J(θ) = -||∇_φ L(φ)||² + λ ||param||²
    - 'virtual_sgd_improve': J(θ)=L_tp_virtual (virtual one-step loss after teleport)
    
    Supports parameterizations for s:
    - 'exp': s = exp(log_s), optimize log_s (default, ensures s > 0)
    - 'direct': s = softplus(u) + eps, optimize u directly
    - 'projected': s directly, projected to be positive after each update
    
    Uses functional_call to search without mutating model.
    
    Args:
        model: Model containing the FFN (must have get_ffn_layers method)
        layer_idx: Index of encoder layer to teleport
        X_full: Input data
        loss_fn: Loss function
        Y: Target data
        lr_theta: Learning rate for symmetry parameter optimization
        steps: Number of optimization steps
        log_s_clip: Clamping range for log_s (used when s_param='exp')
        lr: Step size η for virtual SGD step (only for virtual_loss objective)
        lambda_penalty: Penalty weight λ for regularization
        objective: 'virtual_loss', 'grad_norm', 'bo_grad_norm', or 'virtual_sgd_improve'
        s_param: 'exp', 'direct', or 'projected'
        force_nontrivial_s: If True, rescale log_s direction to hit target magnitude
        theta_target_max_log_s: Target max|log_s| when force_nontrivial_s is enabled
    
    Returns:
        s_best: Best scaling factors (tensor, detached)
        J_before: Objective at identity (float)
        J_best: Best objective value (float)
        diagnostics: Dict with gradient norms, loss/grad_norm before/after, and other metrics
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Run search in eval mode to disable stochastic behavior
    was_training = model.training
    model.eval()
    
    # Save RNG state to ensure teleportation doesn't affect subsequent training
    cpu_rng_start = torch.get_rng_state()
    cuda_rng_start = None
    if torch.cuda.is_available():
        cuda_rng_start = torch.cuda.get_rng_state_all()
    
    try:
        linear1, linear2 = model.get_ffn_layers(layer_idx)
        hidden_dim = linear1.weight.shape[0]
        
        # Store original parameters
        params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}
        
        # Find parameter keys for this FFN dynamically (no hard-coded module path).
        id_to_name = {id(p): name for name, p in model.named_parameters()}
        w1_name = id_to_name.get(id(linear1.weight))
        if w1_name is None:
            raise ValueError(
                f"teleport_ffn_diagonal: layer_idx={layer_idx} missing key for linear1.weight"
            )
        b1_name = None
        if linear1.bias is not None:
            b1_name = id_to_name.get(id(linear1.bias))
            if b1_name is None:
                raise ValueError(
                    f"teleport_ffn_diagonal: layer_idx={layer_idx} missing key for linear1.bias"
                )
        w2_name = id_to_name.get(id(linear2.weight))
        if w2_name is None:
            raise ValueError(
                f"teleport_ffn_diagonal: layer_idx={layer_idx} missing key for linear2.weight"
            )
        
        best_sym_param = None
        best_J = None
        J_before = None
        s_eps = 1e-8
        min_s = None
        max_s = None
        if s_param == 'projected':
            min_s = max(s_eps, math.exp(log_s_clip[0]))
            max_s = math.exp(log_s_clip[1])
        
        # Disable Flash Attention for double backward compatibility
        try:
            from torch.backends.cuda import sdp_kernel
            if torch.cuda.is_available():
                ctx = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            else:
                ctx = contextlib.nullcontext()
        except ImportError:
            ctx = contextlib.nullcontext()
        
        # Choose objective function
        if objective == 'virtual_loss' or objective == 'virtual_sgd_improve':
            if objective == 'virtual_sgd_improve':
                objective_fn = lambda model, params, sym_p, X, Y, loss_fn: objective_virtual_sgd_improve(
                    model, params, sym_p, X, Y, loss_fn, lr, virtual_steps=5, lr_virtual_mult=2.0
                )
            else:
                objective_fn = lambda model, params, sym_p, X, Y, loss_fn: objective_virtual_loss(
                    model, params, sym_p, X, Y, loss_fn, lr, lambda_penalty)
        elif objective == 'grad_norm' or objective == 'bo_grad_norm':
            objective_fn = lambda model, params, sym_p, X, Y, loss_fn: objective_grad_norm(
                model, params, sym_p, X, Y, loss_fn, lambda_penalty)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Measure BEFORE teleportation (at identity transformation)
        with torch.no_grad():
            out_before = functional_call(model, params_dict, (X_full,))
            loss_before = float(loss_fn(out_before, Y).item())
        
        # Compute gradient norm before (on original parameters)
        with torch.enable_grad():
            params_dict_grad = {k: v.clone().requires_grad_(True) for k, v in params_dict.items()}
            params_ordered_grad = OrderedDict(params_dict_grad)
            out_grad = functional_call(model, params_ordered_grad, (X_full,))
            loss_grad = loss_fn(out_grad, Y)
            
            grad_keys = [k for k, v in params_dict_grad.items() if v.requires_grad]
            grad_vals = [params_dict_grad[k] for k in grad_keys]
            grads_before = torch.autograd.grad(loss_grad, grad_vals, create_graph=False, allow_unused=True)
            grad_norm_before = _grad_norm_from_grads(grads_before)
            grad_map_before = {k: g for k, g in zip(grad_keys, grads_before)}
            ffn_names = [w1_name, w2_name]
            if b1_name is not None:
                ffn_names.append(b1_name)
            grad_norm_ffn_before = _grad_norm_from_grad_map(grad_map_before, ffn_names)
            grad_norm_all_before = grad_norm_before
        
        # Compute J_before at identity (s = 1 for all dimensions)
        with ctx:
            if s_param == 'exp':
                sym_param_zero = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            elif s_param == 'direct':
                # For direct param, find u such that softplus(u) ≈ 1
                sym_param_zero = torch.full((hidden_dim,), 0.54, device=device, dtype=dtype, requires_grad=True)
            elif s_param == 'projected':
                # For projected param, s = 1 directly
                sym_param_zero = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            
            s_vec_zero = sym_param_to_s_vec(sym_param_zero, s_param, log_s_clip, s_eps)
            
            params_t_zero = {k: v.clone() for k, v in params_dict.items()}
            params_t_zero[w1_name] = params_dict[w1_name] * s_vec_zero[:, None]
            if b1_name in params_dict and params_dict[b1_name] is not None:
                params_t_zero[b1_name] = params_dict[b1_name] * s_vec_zero
            params_t_zero[w2_name] = params_dict[w2_name] / s_vec_zero[None, :]
            
            params_t_zero_ordered = OrderedDict(params_t_zero)
            J_before = objective_fn(model, params_t_zero_ordered, sym_param_zero, X_full, Y, loss_fn)
            J_before_val = float(J_before.detach().item())
            
            # Compute initial gradient norm at identity
            dJ_dsym_0 = torch.autograd.grad(J_before, sym_param_zero, retain_graph=False, create_graph=False)[0]
        initial_grad_norm = float(dJ_dsym_0.norm().item())
        
        def _grad_norm_sq_from_s_vec(s_vec):
            params_t = {k: v.clone() for k, v in params_dict.items()}
            params_t[w1_name] = params_dict[w1_name] * s_vec[:, None]
            if b1_name in params_dict and params_dict[b1_name] is not None:
                params_t[b1_name] = params_dict[b1_name] * s_vec
            params_t[w2_name] = params_dict[w2_name] / s_vec[None, :]
            
            params_t_grad = {k: v.clone().requires_grad_(True) for k, v in params_t.items()}
            params_t_ordered = OrderedDict(params_t_grad)
            out = functional_call(model, params_t_ordered, (X_full,))
            loss = loss_fn(out, Y)
            grad_keys = [k for k, v in params_t_grad.items() if v.requires_grad]
            grad_vals = [params_t_grad[k] for k in grad_keys]
            grads = torch.autograd.grad(loss, grad_vals, create_graph=False, allow_unused=True)
            total = 0.0
            for g in grads:
                if g is not None:
                    total += (g * g).sum()
            if isinstance(total, torch.Tensor):
                return float(total.item())
            return 0.0
        
        # Initialize for first restart (also used by debug probe below).
        if s_param == 'exp':
            sym_param = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
        elif s_param == 'direct':
            sym_param = torch.full((hidden_dim,), 0.54, device=device, dtype=dtype, requires_grad=True)
        elif s_param == 'projected':
            sym_param = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)
        else:
            raise ValueError(f"Unknown s_param: {s_param}")

        if debug_inner_log:
            print("INNER OPTIMIZATION LOG (seed=0, first teleport attempt)", flush=True)
            print(f"initial ||dJ/dsym_param||: {initial_grad_norm:.6e}", flush=True)
            print("sym_param_optimizer: manual SGD (lr_theta)", flush=True)
            print("sym_param_gradient_clipping: none", flush=True)
            with torch.enable_grad():
                if s_param == 'exp':
                    log_s_plus = sym_param + 0.1
                    log_s_minus = sym_param - 0.1
                    s_vec_plus = torch.exp(log_s_plus)
                    s_vec_minus = torch.exp(log_s_minus)
                elif s_param == 'direct':
                    u_plus = sym_param + 0.1
                    u_minus = sym_param - 0.1
                    s_vec_plus = F.softplus(u_plus) + s_eps
                    s_vec_minus = F.softplus(u_minus) + s_eps
                elif s_param == 'projected':
                    s_vec_plus = (sym_param + 0.05).clamp_min(s_eps)
                    s_vec_minus = (sym_param - 0.05).clamp_min(s_eps)
                else:
                    raise ValueError(f"Unknown s_param: {s_param}")
                grad_norm_sq_plus = _grad_norm_sq_from_s_vec(s_vec_plus)
                grad_norm_sq_minus = _grad_norm_sq_from_s_vec(s_vec_minus)
            print(f"PERTURB + grad_norm_sq: {grad_norm_sq_plus:.6e}", flush=True)
            print(f"PERTURB - grad_norm_sq: {grad_norm_sq_minus:.6e}", flush=True)
            print(
                f"PERTURB + s_min={float(s_vec_plus.min().item()):.6e} "
                f"s_max={float(s_vec_plus.max().item()):.6e}",
                flush=True
            )
            print(
                f"PERTURB - s_min={float(s_vec_minus.min().item()):.6e} "
                f"s_max={float(s_vec_minus.max().item()):.6e}",
                flush=True
            )
        
        max_sym_param_norm = 0.0
        num_restarts = 10 if objective == 'virtual_sgd_improve' else 1
        for restart_idx in range(num_restarts):
            # Exploration for virtual_sgd_improve to avoid identity lock.
            if objective == 'virtual_sgd_improve':
                with torch.no_grad():
                    if s_param == 'exp':
                        init = 5e-2 * torch.randn(hidden_dim, device=device, dtype=dtype)
                    elif s_param == 'direct':
                        init = 0.54 + 5e-2 * torch.randn(hidden_dim, device=device, dtype=dtype)
                    elif s_param == 'projected':
                        init = 1.0 + 5e-2 * torch.randn(hidden_dim, device=device, dtype=dtype)
                        init = init.clamp(min=min_s, max=max_s)
                    else:
                        raise ValueError(f"Unknown s_param: {s_param}")
                sym_param = init.detach().clone().requires_grad_(True)
            elif restart_idx > 0:
                # Non-virtual objective uses only one deterministic start.
                continue

            prev_J_val = None
            for iter_idx in range(steps):
                s_vec = sym_param_to_s_vec(sym_param, s_param, log_s_clip)

                current_norm = float(sym_param.norm().item())
                if current_norm > max_sym_param_norm:
                    max_sym_param_norm = current_norm

                # Create transformed parameter dict
                params_t = {k: v.clone() for k, v in params_dict.items()}
                params_t[w1_name] = params_dict[w1_name] * s_vec[:, None]
                if b1_name in params_dict and params_dict[b1_name] is not None:
                    params_t[b1_name] = params_dict[b1_name] * s_vec
                params_t[w2_name] = params_dict[w2_name] / s_vec[None, :]

                params_t_ordered = OrderedDict(params_t)

                with ctx:
                    J = objective_fn(model, params_t_ordered, sym_param, X_full, Y, loss_fn)

                J_val = float(J.detach().item())
                if debug_inner_log and restart_idx == 0:
                    with torch.enable_grad():
                        grad_norm_sq = _grad_norm_sq_from_s_vec(s_vec)
                    penalty = float((lambda_penalty * (sym_param * sym_param).sum()).item())
                    log_s_cur = torch.log(s_vec.clamp_min(s_eps))
                    max_abs_log_s = float(log_s_cur.abs().max().item())
                    s_min = float(s_vec.min().item())
                    s_max = float(s_vec.max().item())
                    delta_J = float("nan") if prev_J_val is None else (J_val - prev_J_val)
                prev_J_val = J_val

                if best_J is None or J_val < best_J:
                    best_J = J_val
                    best_sym_param = sym_param.detach().clone()

                # Gradient descent on sym_param
                with ctx:
                    dJ_dsym = torch.autograd.grad(J, sym_param, retain_graph=False, create_graph=False)[0]

                sym_param_prev = sym_param.detach().clone()
                with torch.no_grad():
                    sym_param -= lr_theta * dJ_dsym
                    # Project to feasible set for 'projected' parameterization
                    if s_param == 'projected':
                        sym_param.clamp_(min=min_s, max=max_s)
                if debug_inner_log and restart_idx == 0:
                    delta_sym_param_norm = float((sym_param - sym_param_prev).norm().item())
                    print(
                        "INNER_STEP "
                        f"idx={iter_idx} "
                        f"max|log_s|={max_abs_log_s:.6e} "
                        f"s_min={s_min:.6e} "
                        f"s_max={s_max:.6e} "
                        f"grad_norm_sq={grad_norm_sq:.6e} "
                        f"penalty={penalty:.6e} "
                        f"J={J_val:.6e} "
                        f"delta_J={delta_J:.6e} "
                        f"delta_sym_param_norm={delta_sym_param_norm:.6e}",
                        flush=True
                    )
        
        # Convert best_sym_param to s_best
        s_best = sym_param_to_s_vec(best_sym_param, s_param, log_s_clip)
        if debug_inner_log:
            log_s_end = torch.log(s_best.clamp_min(s_eps))
            print(f"INNER_LOOP max|log_s|_end: {float(log_s_end.abs().max().item()):.6e}", flush=True)
        
        # Optional diagnostic forcing of nontrivial scaling (parameterization-agnostic).
        log_s_best = torch.log(s_best.clamp_min(s_eps))
        max_abs_log_s_before_force = float(log_s_best.abs().max().item())
        force_applied = False
        if force_nontrivial_s and max_abs_log_s_before_force < theta_target_max_log_s:
            if max_abs_log_s_before_force > 0.0:
                scale = theta_target_max_log_s / max_abs_log_s_before_force
                log_s_best = log_s_best * scale
            else:
                log_s_best = torch.full_like(log_s_best, theta_target_max_log_s)
            log_s_best = torch.clamp(log_s_best, log_s_clip[0], log_s_clip[1])
            # Regardless of internal parameterization, teleport transform consumes s directly.
            s_best = torch.exp(log_s_best)
            force_applied = True
        
        # Measure AFTER teleportation (apply best transformation to params)
        with torch.no_grad():
            params_after = {k: v.clone() for k, v in params_dict.items()}
            params_after[w1_name] = params_dict[w1_name] * s_best[:, None]
            if b1_name in params_dict and params_dict[b1_name] is not None:
                params_after[b1_name] = params_dict[b1_name] * s_best
            params_after[w2_name] = params_dict[w2_name] / s_best[None, :]
            
            out_after = functional_call(model, params_after, (X_full,))
            loss_after = float(loss_fn(out_after, Y).item())
        
        # Compute gradient norm after (on transformed parameters)
        with torch.enable_grad():
            params_after_grad = {k: v.clone().requires_grad_(True) for k, v in params_after.items()}
            params_after_ordered = OrderedDict(params_after_grad)
            out_grad_after = functional_call(model, params_after_ordered, (X_full,))
            loss_grad_after = loss_fn(out_grad_after, Y)
            
            grad_keys_after = [k for k, v in params_after_grad.items() if v.requires_grad]
            grad_vals_after = [params_after_grad[k] for k in grad_keys_after]
            grads_after = torch.autograd.grad(loss_grad_after, grad_vals_after, create_graph=False, allow_unused=True)
            grad_norm_after = _grad_norm_from_grads(grads_after)
            grad_map_after = {k: g for k, g in zip(grad_keys_after, grads_after)}
            grad_norm_ffn_after = _grad_norm_from_grad_map(grad_map_after, ffn_names)
            grad_norm_all_after = grad_norm_after
        if debug_inner_log:
            print(
                "TELEPORT_GRAD_NORM "
                f"before={grad_norm_before:.6e} "
                f"after={grad_norm_after:.6e} "
                f"delta={float(grad_norm_after - grad_norm_before):.6e}",
                flush=True
            )
        
        # Compute log_s statistics for logging (consistent across parameterizations)
        log_s_best = torch.log(s_best.clamp_min(s_eps))
        
        eps = 1e-12
        delta_grad_norm_all = grad_norm_all_after - grad_norm_all_before
        delta_grad_norm_ffn = grad_norm_ffn_after - grad_norm_ffn_before
        diagnostics = {
            'initial_grad_norm': initial_grad_norm,
            'max_sym_param_norm': max_sym_param_norm,
            'max_log_s_magnitude': float(log_s_best.abs().max().item()),
            'max_abs_log_s': float(log_s_best.abs().max().item()),
            'max_abs_log_s_after_force': float(log_s_best.abs().max().item()),
            'mean_abs_log_s': float(log_s_best.abs().mean().item()),
            's_min': float(s_best.min().item()),
            's_max': float(s_best.max().item()),
            'max_abs_s_minus_one': float((s_best - 1.0).abs().max().item()),
            'force_nontrivial_applied': force_applied,
            'max_abs_log_s_before_force': max_abs_log_s_before_force,
            'force_target_max_log_s': float(theta_target_max_log_s),
            'loss_before': loss_before,
            'loss_after': loss_after,
            'delta_loss': loss_after - loss_before,
            'grad_norm_before': grad_norm_before,
            'grad_norm_after': grad_norm_after,
            'delta_grad_norm': grad_norm_after - grad_norm_before,
            'grad_norm_all_before': grad_norm_all_before,
            'grad_norm_all_after': grad_norm_all_after,
            'delta_grad_norm_all': delta_grad_norm_all,
            'ratio_grad_norm_all': delta_grad_norm_all / max(grad_norm_all_before, eps),
            'grad_norm_ffn_before': grad_norm_ffn_before,
            'grad_norm_ffn_after': grad_norm_ffn_after,
            'delta_grad_norm_ffn': delta_grad_norm_ffn,
            'ratio_grad_norm_ffn': delta_grad_norm_ffn / max(grad_norm_ffn_before, eps)
        }
        
        return s_best, J_before_val, best_J, diagnostics

    finally:
        # Restore RNG state
        torch.set_rng_state(cpu_rng_start)
        if cuda_rng_start is not None:
            torch.cuda.set_rng_state_all(cuda_rng_start)

        # Restore training mode
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Q/K diagonal teleportation
# ---------------------------------------------------------------------------

def _apply_qk_diagonal_inplace(attn, a: torch.Tensor) -> None:
    """
    Apply diagonal Q/K scaling in-place to nn.MultiheadAttention.

    For diagonal a > 0 (shape d = nhead * d_h):
        W_Q' rows i  *= a[i]      (rows 0:d of in_proj_weight)
        W_K' rows i  *= 1/a[i]   (rows d:2*d of in_proj_weight)
        b_Q' *= a,  b_K' *= 1/a  (if bias present)

    Preserves Q_h K_h^T for each head h (exact in arithmetic).
    Uses element-wise ops — no matrix inversion needed.
    """
    d = attn.embed_dim
    a = a.to(device=attn.in_proj_weight.device, dtype=attn.in_proj_weight.dtype)
    a_inv = 1.0 / a
    with torch.no_grad():
        W = attn.in_proj_weight
        W[0:d, :].copy_(W[0:d, :].clone() * a[:, None])
        W[d:2 * d, :].copy_(W[d:2 * d, :].clone() * a_inv[:, None])
        if attn.in_proj_bias is not None:
            b = attn.in_proj_bias
            b[0:d].copy_(b[0:d].clone() * a)
            b[d:2 * d].copy_(b[d:2 * d].clone() * a_inv)


def teleport_qk_diagonal(
    model,
    layer_idx,
    X_full,
    loss_fn,
    Y,
    lr_theta=1e-2,
    steps=20,
    log_a_clip=(-2.0, 2.0),
    lr=0.01,
    restarts=10,
):
    """
    Teleport Q/K projections using diagonal per-head scaling.

    For a > 0 (shape d = nhead * d_h), the transform is:
        W_Q rows i  *= a[i]     (Q projection scaled up)
        W_K rows i  *= 1/a[i]  (K projection scaled down)
        b_Q *= a,  b_K *= 1/a  (if bias present)

    This preserves Q_h K_h^T for each head h up to floating-point rounding.

    Requires model.get_attn_layer(layer_idx) returning nn.MultiheadAttention.

    Uses virtual_sgd_improve objective (K=5 virtual SGD steps) to find
    the scaling that most improves the post-step loss.

    Args:
        model: Model with get_attn_layer(layer_idx) method
        layer_idx: Encoder layer index
        X_full: Input batch
        loss_fn: Loss function
        Y: Target batch
        lr_theta: Learning rate for log_a gradient descent
        steps: Inner optimization steps per restart
        log_a_clip: Clamp range for log_a (same semantics as log_s_clip for FFN)
        lr: SGD step size used in the virtual-step objective
        restarts: Number of random restarts

    Returns:
        a_best: Best diagonal scales (shape d), all positive, detached
        J_before_val: Objective at identity (a = 1)
        J_best_val: Best objective value found
        diagnostics: Dict with log_a statistics and loss info
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    was_training = model.training
    model.eval()

    cpu_rng_start = torch.get_rng_state()
    cuda_rng_start = None
    if torch.cuda.is_available():
        cuda_rng_start = torch.cuda.get_rng_state_all()

    try:
        attn = model.get_attn_layer(layer_idx)
        if attn.in_proj_weight is None:
            raise ValueError(
                "teleport_qk_diagonal requires in_proj_weight "
                "(qkv_same_embed_dim=True)."
            )
        d = attn.embed_dim  # nhead * d_h

        # Find parameter names via identity map
        id_to_name = {id(p): name for name, p in model.named_parameters()}
        proj_weight_name = id_to_name.get(id(attn.in_proj_weight))
        if proj_weight_name is None:
            raise ValueError(
                f"teleport_qk_diagonal: layer_idx={layer_idx} "
                "missing key for in_proj_weight"
            )
        proj_bias_name = None
        if attn.in_proj_bias is not None:
            proj_bias_name = id_to_name.get(id(attn.in_proj_bias))
            if proj_bias_name is None:
                raise ValueError(
                    f"teleport_qk_diagonal: layer_idx={layer_idx} "
                    "missing key for in_proj_bias"
                )

        # Detached snapshot of all parameters for functional_call
        params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}

        # Measure loss at identity before any transform
        with torch.no_grad():
            out_before = functional_call(model, params_dict, (X_full,))
            loss_before = float(loss_fn(out_before, Y).item())

        def _build_qk_params(sym_param):
            """Build transformed param dict given log_a = sym_param (has grad)."""
            a = sym_param_to_s_vec(sym_param, 'exp', log_a_clip)  # shape (d,)
            a_inv = 1.0 / a

            params_t = {k: v.clone() for k, v in params_dict.items()}
            W = params_dict[proj_weight_name]   # (3*d, d_model), detached
            W_Q = W[0:d, :]
            W_K = W[d:2 * d, :]
            W_V = W[2 * d:, :]
            # W_Q * a and W_K * a_inv have grad_fn because a depends on sym_param
            params_t[proj_weight_name] = torch.cat([
                W_Q * a[:, None],
                W_K * a_inv[:, None],
                W_V,
            ], dim=0)

            if proj_bias_name is not None:
                b = params_dict[proj_bias_name]  # (3*d,), detached
                params_t[proj_bias_name] = torch.cat([
                    b[0:d] * a,
                    b[d:2 * d] * a_inv,
                    b[2 * d:],
                ], dim=0)

            return params_t, a

        # J at identity (sym_param = 0 → a = 1 everywhere)
        # Force math SDPA kernel: in_proj_weight has requires_grad=True (via
        # sym_param), so flash-attention-for-cpu cannot handle create_graph=True.
        with _math_sdpa():
            sym_param_zero = torch.zeros(d, device=device, dtype=dtype, requires_grad=True)
            params_t_zero, _ = _build_qk_params(sym_param_zero)
            J_before_obj = objective_virtual_sgd_improve(
                model, OrderedDict(params_t_zero), sym_param_zero,
                X_full, Y, loss_fn, lr, virtual_steps=5, lr_virtual_mult=2.0,
            )
            J_before_val = float(J_before_obj.detach().item())

            best_sym_param = None
            best_J = None

            for _ in range(restarts):
                with torch.no_grad():
                    init = 5e-2 * torch.randn(d, device=device, dtype=dtype)
                sym_param = init.detach().clone().requires_grad_(True)

                for _ in range(steps):
                    params_t, _ = _build_qk_params(sym_param)
                    J = objective_virtual_sgd_improve(
                        model, OrderedDict(params_t), sym_param,
                        X_full, Y, loss_fn, lr, virtual_steps=5, lr_virtual_mult=2.0,
                    )
                    J_val = float(J.detach().item())
                    if best_J is None or J_val < best_J:
                        best_J = J_val
                        best_sym_param = sym_param.detach().clone()

                    dJ = torch.autograd.grad(J, sym_param, retain_graph=False)[0]
                    with torch.no_grad():
                        sym_param = (sym_param - lr_theta * dJ)
                    sym_param = sym_param.detach().requires_grad_(True)

        if best_sym_param is None:
            best_sym_param = torch.zeros(d, device=device, dtype=dtype)

        a_best = sym_param_to_s_vec(best_sym_param, 'exp', log_a_clip).detach()

        # Measure loss at best transform (stateless)
        with torch.no_grad():
            a_inv_best = 1.0 / a_best
            W = params_dict[proj_weight_name]
            params_after = {k: v.clone() for k, v in params_dict.items()}
            params_after[proj_weight_name] = torch.cat([
                W[0:d, :] * a_best[:, None],
                W[d:2 * d, :] * a_inv_best[:, None],
                W[2 * d:, :],
            ], dim=0)
            if proj_bias_name is not None:
                b = params_dict[proj_bias_name]
                params_after[proj_bias_name] = torch.cat([
                    b[0:d] * a_best,
                    b[d:2 * d] * a_inv_best,
                    b[2 * d:],
                ], dim=0)
            out_after = functional_call(model, params_after, (X_full,))
            loss_after = float(loss_fn(out_after, Y).item())

        log_a_best = torch.log(a_best.clamp_min(1e-12))
        diagnostics = {
            'max_abs_log_a': float(log_a_best.abs().max().item()),
            'mean_abs_log_a': float(log_a_best.abs().mean().item()),
            'a_min': float(a_best.min().item()),
            'a_max': float(a_best.max().item()),
            'max_abs_a_minus_one': float((a_best - 1.0).abs().max().item()),
            'loss_before': loss_before,
            'loss_after': loss_after,
            'delta_loss': loss_after - loss_before,
        }

        return a_best, J_before_val, best_J if best_J is not None else J_before_val, diagnostics

    finally:
        torch.set_rng_state(cpu_rng_start)
        if cuda_rng_start is not None:
            torch.cuda.set_rng_state_all(cuda_rng_start)
        if was_training:
            model.train()

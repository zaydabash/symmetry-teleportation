#!/usr/bin/env python3
"""
Minimal Reproducible Benchmark: SGD vs SGD+Teleportation on Tiny Transformer

Hard requirements:
- One command reproduces everything
- Paired runs (same seed, same data, same init)
- Outputs: results/curve.png, results/summary.json
- Runs in < 2-3 minutes on CPU (default settings)
- 20 seeds by default

Usage:
    python scripts/bench_tiny_transformer.py
    python scripts/bench_tiny_transformer.py --seeds 0 1 2 3 4 --steps 200
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import random
import numpy as np
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import TinyTransformer from existing implementation
from transformer_teleport_optimizer import TinyTransformer
# Import TeleportSGD from packaged module
from symmetry_teleport import TeleportSGD


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_dataset(n_samples=200, seq_len=8, feature_dim=4, seed=42):
    """
    Create a fixed synthetic dataset for regression.
    Task: Predict mean of input features.
    Uses a local torch.Generator so the global RNG is NOT touched.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(n_samples, seq_len, feature_dim, generator=g)
    # Target: mean of features across sequence
    Y = X.mean(dim=(1, 2)).unsqueeze(1).expand(-1, feature_dim)
    return X, Y


def _tensor_md5(t: torch.Tensor) -> str:
    """MD5 hash of a tensor's bytes (detached, cpu, contiguous)."""
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def get_ffn_params(model, layer_idx=0):
    """Return FFN parameters affected by diagonal symmetry for a layer."""
    linear1, linear2 = model.get_ffn_layers(layer_idx)
    params = [linear1.weight, linear1.bias, linear2.weight]
    return [p for p in params if p is not None]


def grad_norm_from_params(params):
    """Compute L2 grad norm over a set of parameters."""
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += (p.grad.detach() * p.grad.detach()).sum().item()
    return total ** 0.5


def model_config_from_budget(param_budget):
    """Return TinyTransformer kwargs for requested parameter budget."""
    if param_budget == 'tiny':
        # ~676 params: fits requested tiny budget (200-800)
        return dict(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    if param_budget == 'small':
        # Existing benchmark configuration
        return dict(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0)
    raise ValueError(f"Unknown param_budget: {param_budget}")


def run_single_seed(
    seed,
    steps,
    batch_size,
    lr,
    teleport_every,
    lambda_penalty,
    inner_steps,
    log_s_clip,
    lr_theta=None,
    objective='virtual_loss',
    s_param='exp',
    force_nontrivial_s=False,
    theta_target_max_log_s=0.5,
    rescale_lr_post_teleport=False,
    model_kwargs=None,
    device='cpu'
):
    """
    Run paired baseline and teleport training for one seed.
    Returns: (baseline_losses, teleport_losses, active_events, init_param_sample, first_batch_mean)
    """
    if lr_theta is None:
        lr_theta = lr
    if model_kwargs is None:
        model_kwargs = model_config_from_budget('small')

    # Create dataset using local Generator (does NOT touch global RNG)
    X_full, Y_full = create_synthetic_dataset(n_samples=200, seq_len=8, feature_dim=4, seed=seed)
    X_full = X_full.to(device)
    Y_full = Y_full.to(device)

    # Seed global RNG immediately before model creation so init is tied to seed
    set_seed(seed)

    # Create model
    model = TinyTransformer(**model_kwargs)
    model = model.to(device)

    # Save initial state
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Hash-based diversity diagnostics
    data_hash = _tensor_md5(X_full[:2])
    param_vec = torch.cat([p.detach().flatten()[:1000].cpu() for p in model.parameters()])
    param_hash = hashlib.md5(param_vec.numpy().tobytes()).hexdigest()

    # Legacy scalars (kept for backward compat)
    init_param_sample = float(model.embed.weight[0, 0].item())
    first_batch_indices = torch.arange(0, batch_size) % len(X_full)
    first_batch_mean = float(X_full[first_batch_indices].mean().item())

    # One-line debug print per seed (before training)
    print(f"seed={seed} data_hash={data_hash} param_hash={param_hash} rand_probe={float(torch.rand(1).item()):.6f}")
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # ============================================================================
    # Baseline: Plain SGD
    # ============================================================================
    set_seed(seed)  # Reset seed before baseline
    model.load_state_dict(init_state)
    model.train()
    
    optimizer_baseline = torch.optim.SGD(model.parameters(), lr=lr)
    baseline_losses = []
    baseline_grad_norm_all = []
    baseline_grad_norm_ffn = []
    ffn_params = get_ffn_params(model, layer_idx=0)
    
    for step_idx in range(steps):
        # Sample minibatch deterministically (fixed order based on step)
        indices = torch.arange(step_idx * batch_size, (step_idx + 1) * batch_size) % len(X_full)
        X_batch = X_full[indices]
        Y_batch = Y_full[indices]
        
        optimizer_baseline.zero_grad()
        out = model(X_batch)
        loss = loss_fn(out, Y_batch)
        loss.backward()
        
        baseline_grad_norm_all.append(grad_norm_from_params(model.parameters()))
        baseline_grad_norm_ffn.append(grad_norm_from_params(ffn_params))
        optimizer_baseline.step()
        
        baseline_losses.append(float(loss.item()))
    
    # ============================================================================
    # Teleport: SGD + Teleportation
    # ============================================================================
    set_seed(seed)  # Reset seed before teleport
    model.load_state_dict(init_state)
    model.train()
    
    # Create teleport optimizer using packaged TeleportSGD
    optimizer_teleport = TeleportSGD(
        params=model.parameters(),
        lr=lr,
        teleport_every=teleport_every,
        teleport_config={
            'model': model,
            'layer_idx': 0,
            'X_teleport': X_full,
            'Y_teleport': Y_full,
            'loss_fn': loss_fn,
            'lr_theta': lr_theta,
            'inner_steps': inner_steps,
            'lambda_penalty': lambda_penalty,
            'log_s_clip': log_s_clip,
            'objective': objective,
            's_param': s_param,
            'force_nontrivial_s': force_nontrivial_s,
            'theta_target_max_log_s': theta_target_max_log_s,
            'rescale_lr_post_teleport': bool(rescale_lr_post_teleport),
            'debug_first_attempt': seed == 0
        }
    )
    
    teleport_losses = []
    teleport_grad_norm_all = []
    teleport_grad_norm_ffn = []
    
    for step_idx in range(steps):
        # Sample minibatch deterministically (same as baseline)
        indices = torch.arange(step_idx * batch_size, (step_idx + 1) * batch_size) % len(X_full)
        X_batch = X_full[indices]
        Y_batch = Y_full[indices]
        
        optimizer_teleport.zero_grad()
        out = model(X_batch)
        loss = loss_fn(out, Y_batch)
        loss.backward()
        
        teleport_grad_norm_all.append(grad_norm_from_params(model.parameters()))
        teleport_grad_norm_ffn.append(grad_norm_from_params(ffn_params))
        optimizer_teleport.step()  # This automatically applies teleportation when scheduled
        
        teleport_losses.append(float(loss.item()))
    
    # Get active events from optimizer (strict criteria)
    active_events = len(optimizer_teleport.teleport_active_steps)
    
    # Get detailed teleport attempt data
    teleport_attempts = optimizer_teleport.teleport_attempts
    
    return (
        baseline_losses,
        teleport_losses,
        active_events,
        init_param_sample,
        first_batch_mean,
        teleport_attempts,
        baseline_grad_norm_all,
        baseline_grad_norm_ffn,
        teleport_grad_norm_all,
        teleport_grad_norm_ffn,
        data_hash,
        param_hash,
    )


def compute_steps_to_threshold(losses, threshold):
    """
    Compute number of steps to reach threshold.
    Returns: step index or None if never reached.
    """
    for i, loss in enumerate(losses):
        if loss <= threshold:
            return i
    return None


def aggregate_results(all_results, config):
    """
    Aggregate results across seeds.
    Returns: summary dict with curves and metrics.
    """
    seeds = sorted(all_results.keys())
    n_seeds = len(seeds)
    
    # Extract losses
    baseline_curves = np.array([all_results[s]['baseline_losses'] for s in seeds])
    teleport_curves = np.array([all_results[s]['teleport_losses'] for s in seeds])
    
    # Extract grad norm curves
    baseline_grad_norm_all_curves = np.array([
        all_results[s].get('baseline_grad_norm_all', [0.0] * len(all_results[s]['baseline_losses']))
        for s in seeds
    ])
    baseline_grad_norm_ffn_curves = np.array([
        all_results[s].get('baseline_grad_norm_ffn', [0.0] * len(all_results[s]['baseline_losses']))
        for s in seeds
    ])
    teleport_grad_norm_all_curves = np.array([
        all_results[s].get('teleport_grad_norm_all', [0.0] * len(all_results[s]['teleport_losses']))
        for s in seeds
    ])
    teleport_grad_norm_ffn_curves = np.array([
        all_results[s].get('teleport_grad_norm_ffn', [0.0] * len(all_results[s]['teleport_losses']))
        for s in seeds
    ])
    
    # Compute mean/std curves
    baseline_mean = baseline_curves.mean(axis=0).tolist()
    baseline_std = baseline_curves.std(axis=0).tolist()
    teleport_mean = teleport_curves.mean(axis=0).tolist()
    teleport_std = teleport_curves.std(axis=0).tolist()
    
    baseline_grad_norm_all_mean = baseline_grad_norm_all_curves.mean(axis=0).tolist()
    baseline_grad_norm_all_std = baseline_grad_norm_all_curves.std(axis=0).tolist()
    baseline_grad_norm_ffn_mean = baseline_grad_norm_ffn_curves.mean(axis=0).tolist()
    baseline_grad_norm_ffn_std = baseline_grad_norm_ffn_curves.std(axis=0).tolist()
    teleport_grad_norm_all_mean = teleport_grad_norm_all_curves.mean(axis=0).tolist()
    teleport_grad_norm_all_std = teleport_grad_norm_all_curves.std(axis=0).tolist()
    teleport_grad_norm_ffn_mean = teleport_grad_norm_ffn_curves.mean(axis=0).tolist()
    teleport_grad_norm_ffn_std = teleport_grad_norm_ffn_curves.std(axis=0).tolist()
    
    # Compute final loss stats
    baseline_final = baseline_curves[:, -1]
    teleport_final = teleport_curves[:, -1]

    # Loss-at-step-k report
    loss_at_step_k = {}
    for k in [25, 40, 75, 150, 500]:
        if baseline_curves.shape[1] >= k:
            idx = k - 1
            b_vals = baseline_curves[:, idx]
            t_vals = teleport_curves[:, idx]
            loss_at_step_k[str(k)] = {
                'baseline_mean': float(np.mean(b_vals)),
                'baseline_std': float(np.std(b_vals)),
                'teleport_mean': float(np.mean(t_vals)),
                'teleport_std': float(np.std(t_vals)),
            }

    # Area under curve (AUC) over full run
    baseline_auc = np.trapz(baseline_curves, dx=1.0, axis=1)
    teleport_auc = np.trapz(teleport_curves, dx=1.0, axis=1)
    area_under_curve = {
        'baseline_mean': float(np.mean(baseline_auc)),
        'baseline_std': float(np.std(baseline_auc)),
        'teleport_mean': float(np.mean(teleport_auc)),
        'teleport_std': float(np.std(teleport_auc)),
    }
    paired_auc_delta = teleport_auc - baseline_auc
    area_under_curve['paired_delta_mean'] = float(np.mean(paired_auc_delta))
    area_under_curve['paired_delta_std'] = float(np.std(paired_auc_delta))
    area_under_curve['paired_delta_values'] = paired_auc_delta.tolist()

    # Fixed-threshold speedup report
    speedup_report = {}
    for thr in [0.05, 0.02, 0.01]:
        b_steps = []
        t_steps = []
        for s in seeds:
            b_losses = all_results[s]['baseline_losses']
            t_losses = all_results[s]['teleport_losses']
            b_step = compute_steps_to_threshold(b_losses, thr)
            t_step = compute_steps_to_threshold(t_losses, thr)
            if b_step is None:
                b_step = len(b_losses)
            if t_step is None:
                t_step = len(t_losses)
            b_steps.append(b_step)
            t_steps.append(t_step)
        b_mean = float(np.mean(b_steps))
        t_mean = float(np.mean(t_steps))
        speedup_pct = (b_mean - t_mean) / b_mean * 100 if b_mean > 0 else 0.0
        speedup_report[f"{thr:.2f}"] = {
            'baseline_mean': b_mean,
            'baseline_std': float(np.std(b_steps)),
            'teleport_mean': t_mean,
            'teleport_std': float(np.std(t_steps)),
            'speedup_percent': speedup_pct,
        }
    
    # Choose thresholds based on convergence progress (not final loss percentiles)
    # Compute from baseline curve: fraction of distance from initial to final
    baseline_initial = baseline_mean[0]
    baseline_final_mean = baseline_mean[-1]
    
    # Threshold A: 70% of the way from initial to final
    threshold_A = baseline_final_mean + 0.30 * (baseline_initial - baseline_final_mean)
    # Threshold B: 90% of the way from initial to final
    threshold_B = baseline_final_mean + 0.10 * (baseline_initial - baseline_final_mean)
    
    # Compute steps-to-threshold for each seed
    baseline_steps_A = []
    baseline_steps_B = []
    teleport_steps_A = []
    teleport_steps_B = []
    
    for s in seeds:
        b_losses = all_results[s]['baseline_losses']
        t_losses = all_results[s]['teleport_losses']
        
        b_step_A = compute_steps_to_threshold(b_losses, threshold_A)
        b_step_B = compute_steps_to_threshold(b_losses, threshold_B)
        t_step_A = compute_steps_to_threshold(t_losses, threshold_A)
        t_step_B = compute_steps_to_threshold(t_losses, threshold_B)
        
        if b_step_A is not None:
            baseline_steps_A.append(b_step_A)
        if b_step_B is not None:
            baseline_steps_B.append(b_step_B)
        if t_step_A is not None:
            teleport_steps_A.append(t_step_A)
        if t_step_B is not None:
            teleport_steps_B.append(t_step_B)
    
    # Compute mean/std for steps-to-threshold (if any seeds reached it)
    if len(baseline_steps_A) > 0 and len(teleport_steps_A) > 0:
        baseline_steps_A_mean = float(np.mean(baseline_steps_A))
        baseline_steps_A_std = float(np.std(baseline_steps_A))
        teleport_steps_A_mean = float(np.mean(teleport_steps_A))
        teleport_steps_A_std = float(np.std(teleport_steps_A))
        speedup_A = (baseline_steps_A_mean - teleport_steps_A_mean) / baseline_steps_A_mean * 100 if baseline_steps_A_mean > 0 else 0.0
    else:
        baseline_steps_A_mean = baseline_steps_A_std = None
        teleport_steps_A_mean = teleport_steps_A_std = None
        speedup_A = None
    
    if len(baseline_steps_B) > 0 and len(teleport_steps_B) > 0:
        baseline_steps_B_mean = float(np.mean(baseline_steps_B))
        baseline_steps_B_std = float(np.std(baseline_steps_B))
        teleport_steps_B_mean = float(np.mean(teleport_steps_B))
        teleport_steps_B_std = float(np.std(teleport_steps_B))
        speedup_B = (baseline_steps_B_mean - teleport_steps_B_mean) / baseline_steps_B_mean * 100 if baseline_steps_B_mean > 0 else 0.0
    else:
        baseline_steps_B_mean = baseline_steps_B_std = None
        teleport_steps_B_mean = teleport_steps_B_std = None
        speedup_B = None
    
    # Active rate stats (strict criteria)
    active_by_seed = [all_results[s]['active_events'] for s in seeds]
    if config['teleport_every'] > 0:
        total_teleport_opportunities = [len(all_results[s]['teleport_losses']) // config['teleport_every'] for s in seeds]
        active_rate_by_seed = [active_by_seed[i] / max(total_teleport_opportunities[i], 1) * 100 for i in range(n_seeds)]
        active_rate_mean = float(np.mean(active_rate_by_seed))
    else:
        total_teleport_opportunities = [0] * n_seeds
        active_rate_by_seed = [0.0] * n_seeds
        active_rate_mean = 0.0
    
    # Collect all teleport attempts across seeds
    all_attempts = []
    for s in seeds:
        all_attempts.extend(all_results[s].get('teleport_attempts', []))
    
    # Compute statistics on attempts
    accepted_attempts = [a for a in all_attempts if a.get('accepted', False)]
    nontrivial_attempts = [a for a in all_attempts if a.get('nontrivial', False)]
    active_strict_attempts = [a for a in all_attempts if a.get('active_strict', False)]
    
    teleport_stats = {
        'attempt_count': len(all_attempts),
        'total_attempts': len(all_attempts),  # Keep for backward compat
        'accepted_count': len(accepted_attempts),
        'accepted_rate': 100.0 * len(accepted_attempts) / max(len(all_attempts), 1),
        'nontrivial_count': len(nontrivial_attempts),
        'nontrivial_rate': 100.0 * len(nontrivial_attempts) / max(len(all_attempts), 1),
        'active_count_strict': len(active_strict_attempts),
        'active_count': len(active_strict_attempts),  # Keep for backward compat
        'active_rate_strict': 100.0 * len(active_strict_attempts) / max(len(all_attempts), 1),
        'acceptance_rate': 100.0 * len(accepted_attempts) / max(len(all_attempts), 1)  # Keep for backward compat
    }
    
    # Statistics on accepted attempts
    if accepted_attempts:
        delta_Js = [a['delta_J'] for a in accepted_attempts]
        max_abs_log_s_vals = [a['max_abs_log_s'] for a in accepted_attempts]
        mean_abs_log_s_vals = [a['mean_abs_log_s'] for a in accepted_attempts]
        
        teleport_stats['accepted_delta_J_mean'] = float(np.mean(delta_Js))
        teleport_stats['accepted_delta_J_std'] = float(np.std(delta_Js))
        teleport_stats['accepted_delta_J_min'] = float(np.min(delta_Js))
        teleport_stats['accepted_max_abs_log_s_mean'] = float(np.mean(max_abs_log_s_vals))
        teleport_stats['accepted_max_abs_log_s_std'] = float(np.std(max_abs_log_s_vals))
        teleport_stats['accepted_mean_abs_log_s_mean'] = float(np.mean(mean_abs_log_s_vals))
    else:
        teleport_stats['accepted_delta_J_mean'] = None
        teleport_stats['accepted_delta_J_std'] = None
        teleport_stats['accepted_delta_J_min'] = None
        teleport_stats['accepted_max_abs_log_s_mean'] = None
        teleport_stats['accepted_max_abs_log_s_std'] = None
        teleport_stats['accepted_mean_abs_log_s_mean'] = None
    
    # Before/after teleport invariance statistics (Bo's request)
    if all_attempts:
        delta_loss_vals = [a.get('delta_loss_teleport', 0.0) for a in all_attempts]
        delta_grad_norm_vals = [a.get('delta_grad_norm_teleport', 0.0) for a in all_attempts]
        loss_before_vals = [a.get('loss_before_teleport', 0.0) for a in all_attempts]
        grad_norm_before_vals = [a.get('grad_norm_before_teleport', 0.0) for a in all_attempts]
        grad_norm_after_vals = [a.get('grad_norm_after_teleport', 0.0) for a in all_attempts]
        
        delta_grad_norm_all_vals = [a.get('delta_grad_norm_all_teleport', 0.0) for a in all_attempts]
        ratio_grad_norm_all_vals = [a.get('ratio_grad_norm_all_teleport', 0.0) for a in all_attempts]
        delta_grad_norm_ffn_vals = [a.get('delta_grad_norm_ffn_teleport', 0.0) for a in all_attempts]
        ratio_grad_norm_ffn_vals = [a.get('ratio_grad_norm_ffn_teleport', 0.0) for a in all_attempts]
        delta_virtual_vals = [a.get('delta_virtual') for a in all_attempts if a.get('delta_virtual') is not None]
        L_baseline_virtual_vals = [a.get('L_baseline_virtual') for a in all_attempts if a.get('L_baseline_virtual') is not None]
        L_tp_virtual_vals = [a.get('L_tp_virtual') for a in all_attempts if a.get('L_tp_virtual') is not None]
        grad_norm_all_before_vals = [a.get('grad_norm_all_before_teleport', 0.0) for a in all_attempts]
        grad_norm_ffn_before_vals = [a.get('grad_norm_ffn_before_teleport', 0.0) for a in all_attempts]
        
        max_abs_log_s_vals_all = [a.get('max_abs_log_s', 0.0) for a in all_attempts]
        mean_abs_log_s_vals_all = [a.get('mean_abs_log_s', 0.0) for a in all_attempts]
        s_min_vals = [a.get('s_min', 0.0) for a in all_attempts]
        s_max_vals = [a.get('s_max', 0.0) for a in all_attempts]
        max_abs_s_minus_one_vals = [a.get('max_abs_s_minus_one', 0.0) for a in all_attempts]
        delta_J_vals_all = [a.get('delta_J', 0.0) for a in all_attempts]
        
        teleport_stats['delta_loss_mean'] = float(np.mean(delta_loss_vals))
        teleport_stats['delta_loss_std'] = float(np.std(delta_loss_vals))
        teleport_stats['delta_loss_max_abs'] = float(np.max(np.abs(delta_loss_vals)))
        teleport_stats['delta_grad_norm_mean'] = float(np.mean(delta_grad_norm_vals))
        teleport_stats['delta_grad_norm_std'] = float(np.std(delta_grad_norm_vals))
        teleport_stats['delta_grad_norm_median'] = float(np.median(delta_grad_norm_vals))
        teleport_stats['loss_before_mean'] = float(np.mean(loss_before_vals))
        teleport_stats['grad_norm_before_mean'] = float(np.mean(grad_norm_before_vals))
        teleport_stats['grad_norm_after_mean'] = float(np.mean(grad_norm_after_vals))
        teleport_stats['delta_grad_norm_all_median'] = float(np.median(delta_grad_norm_all_vals))
        teleport_stats['ratio_grad_norm_all_median'] = float(np.median(ratio_grad_norm_all_vals))
        teleport_stats['delta_grad_norm_ffn_median'] = float(np.median(delta_grad_norm_ffn_vals))
        teleport_stats['ratio_grad_norm_ffn_median'] = float(np.median(ratio_grad_norm_ffn_vals))
        teleport_stats['grad_norm_all_before_median'] = float(np.median(grad_norm_all_before_vals))
        teleport_stats['grad_norm_ffn_before_median'] = float(np.median(grad_norm_ffn_before_vals))
        if delta_virtual_vals:
            teleport_stats['delta_virtual_mean'] = float(np.mean(delta_virtual_vals))
            teleport_stats['delta_virtual_median'] = float(np.median(delta_virtual_vals))
            teleport_stats['delta_virtual_min'] = float(np.min(delta_virtual_vals))
            teleport_stats['delta_virtual_max'] = float(np.max(delta_virtual_vals))
            teleport_stats['L_baseline_virtual_mean'] = float(np.mean(L_baseline_virtual_vals))
            teleport_stats['L_tp_virtual_mean'] = float(np.mean(L_tp_virtual_vals))
        else:
            teleport_stats['delta_virtual_mean'] = None
            teleport_stats['delta_virtual_median'] = None
            teleport_stats['delta_virtual_min'] = None
            teleport_stats['delta_virtual_max'] = None
            teleport_stats['L_baseline_virtual_mean'] = None
            teleport_stats['L_tp_virtual_mean'] = None
        teleport_stats['max_abs_log_s_mean'] = float(np.mean(max_abs_log_s_vals_all))
        teleport_stats['max_abs_log_s_median'] = float(np.median(max_abs_log_s_vals_all))
        teleport_stats['mean_abs_log_s_mean'] = float(np.mean(mean_abs_log_s_vals_all))
        teleport_stats['s_min_mean'] = float(np.mean(s_min_vals))
        teleport_stats['s_max_mean'] = float(np.mean(s_max_vals))
        teleport_stats['max_abs_s_minus_one_mean'] = float(np.mean(max_abs_s_minus_one_vals))
        teleport_stats['delta_J_mean_all'] = float(np.mean(delta_J_vals_all))
        teleport_stats['delta_J_median_all'] = float(np.median(delta_J_vals_all))
    else:
        teleport_stats['delta_loss_mean'] = None
        teleport_stats['delta_loss_std'] = None
        teleport_stats['delta_loss_max_abs'] = None
        teleport_stats['delta_grad_norm_mean'] = None
        teleport_stats['delta_grad_norm_std'] = None
        teleport_stats['delta_grad_norm_median'] = None
        teleport_stats['loss_before_mean'] = None
        teleport_stats['grad_norm_before_mean'] = None
        teleport_stats['grad_norm_after_mean'] = None
        teleport_stats['delta_grad_norm_all_median'] = None
        teleport_stats['ratio_grad_norm_all_median'] = None
        teleport_stats['delta_grad_norm_ffn_median'] = None
        teleport_stats['ratio_grad_norm_ffn_median'] = None
        teleport_stats['grad_norm_all_before_median'] = None
        teleport_stats['grad_norm_ffn_before_median'] = None
        teleport_stats['delta_virtual_mean'] = None
        teleport_stats['delta_virtual_median'] = None
        teleport_stats['delta_virtual_min'] = None
        teleport_stats['delta_virtual_max'] = None
        teleport_stats['L_baseline_virtual_mean'] = None
        teleport_stats['L_tp_virtual_mean'] = None
        teleport_stats['max_abs_log_s_mean'] = None
        teleport_stats['max_abs_log_s_median'] = None
        teleport_stats['mean_abs_log_s_mean'] = None
        teleport_stats['s_min_mean'] = None
        teleport_stats['s_max_mean'] = None
        teleport_stats['max_abs_s_minus_one_mean'] = None
        teleport_stats['delta_J_mean_all'] = None
        teleport_stats['delta_J_median_all'] = None
    
    # Seed diversity stats (hash-based)
    data_hashes = [all_results[s].get('data_hash', '') for s in seeds]
    param_hashes = [all_results[s].get('param_hash', '') for s in seeds]
    unique_data_hashes = len(set(data_hashes))
    unique_param_hashes = len(set(param_hashes))
    # Legacy scalars kept for backward compat
    init_params = [all_results[s]['init_param_sample'] for s in seeds]
    first_batches = [all_results[s]['first_batch_mean'] for s in seeds]
    init_param_std = float(np.std(init_params))
    first_batch_std = float(np.std(first_batches))

    summary = {
        'config': config,
        'seeds': seeds,
        'curve': {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'teleport_mean': teleport_mean,
            'teleport_std': teleport_std
        },
        'grad_norm_curves': {
            'baseline_all_mean': baseline_grad_norm_all_mean,
            'baseline_all_std': baseline_grad_norm_all_std,
            'baseline_ffn_mean': baseline_grad_norm_ffn_mean,
            'baseline_ffn_std': baseline_grad_norm_ffn_std,
            'teleport_all_mean': teleport_grad_norm_all_mean,
            'teleport_all_std': teleport_grad_norm_all_std,
            'teleport_ffn_mean': teleport_grad_norm_ffn_mean,
            'teleport_ffn_std': teleport_grad_norm_ffn_std
        },
        'seed_diversity': {
            'init_param_std': init_param_std,
            'first_batch_std': first_batch_std,
            'unique_data_hashes': unique_data_hashes,
            'unique_param_hashes': unique_param_hashes,
            'data_hashes': data_hashes,
            'param_hashes': param_hashes,
        },
        'teleport_stats': teleport_stats,
        'metrics': {
            'final_loss_mean': {
                'baseline': float(np.mean(baseline_final)),
                'baseline_std': float(np.std(baseline_final)),
                'teleport': float(np.mean(teleport_final)),
                'teleport_std': float(np.std(teleport_final))
            },
            'threshold_A': threshold_A,
            'threshold_B': threshold_B,
            'steps_to_threshold_A': {
                'baseline_mean': baseline_steps_A_mean,
                'baseline_std': baseline_steps_A_std,
                'teleport_mean': teleport_steps_A_mean,
                'teleport_std': teleport_steps_A_std
            },
            'steps_to_threshold_B': {
                'baseline_mean': baseline_steps_B_mean,
                'baseline_std': baseline_steps_B_std,
                'teleport_mean': teleport_steps_B_mean,
                'teleport_std': teleport_steps_B_std
            },
            'loss_at_step_k': loss_at_step_k,
            'area_under_curve': area_under_curve,
            'speedup_report_fixed_thresholds': speedup_report,
            'speedup_A_percent': speedup_A,
            'speedup_B_percent': speedup_B,
            'active_rate_mean': active_rate_mean,
            'active_rate_by_seed': active_rate_by_seed
        }
    }
    
    return summary


def _series_digest(arr):
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    return digest


def plot_curves(summary, outpath):
    """
    Generate curve plot and save to outpath.
    """
    curve = summary['curve']
    config = summary['config']
    
    baseline_mean = np.array(curve['baseline_mean'])
    teleport_mean = np.array(curve['teleport_mean'])
    steps = np.arange(len(baseline_mean))
    
    baseline_hash = _series_digest(baseline_mean)
    teleport_hash = _series_digest(teleport_mean)
    baseline_first5 = baseline_mean[:5].tolist()
    teleport_first5 = teleport_mean[:5].tolist()
    series_allclose = bool(np.allclose(baseline_mean, teleport_mean))
    
    metadata = {
        'baseline_hash_md5': baseline_hash,
        'teleport_hash_md5': teleport_hash,
        'baseline_first5': baseline_first5,
        'teleport_first5': teleport_first5,
        'series_allclose': series_allclose
    }
    meta_path = outpath.with_name('plot_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Plot metadata:")
    print(f"  baseline_hash_md5: {baseline_hash}")
    print(f"  teleport_hash_md5: {teleport_hash}")
    print(f"  baseline_first5: {baseline_first5}")
    print(f"  teleport_first5: {teleport_first5}")
    print(f"  series_allclose: {series_allclose}")
    if series_allclose:
        print("  SERIES IDENTICAL: check logging")
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, baseline_mean, label='Baseline SGD', linewidth=2, color='blue')
    plt.plot(steps, teleport_mean, label='SGD + Teleport', linewidth=2, color='orange')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f"Tiny Transformer Benchmark ({len(summary['seeds'])} seeds, lr={config['lr']}, teleport_every={config['teleport_every']})", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved plot: {outpath}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark SGD vs SGD+Teleportation')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(20)), help='Seeds to run (default: 0-19)')
    parser.add_argument('--steps', type=int, default=300, help='Training steps per seed')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--param-budget', type=str, default='small', choices=['tiny', 'small'], help='Model size preset: tiny (~200-800 params) or small (default)')
    parser.add_argument('--teleport-every', type=int, default=20, help='Teleport every N steps')
    parser.add_argument('--lambda-penalty', type=float, default=1e-3, help='Penalty weight for log_s')
    parser.add_argument('--inner-steps', type=int, default=20, help='Inner optimization steps for teleportation')
    parser.add_argument('--log-s-clip', type=str, default='-2,2', help='log_s clipping range (min,max)')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive teleportation settings (lambda=1e-6, inner_steps=50, lr_theta=0.1)')
    parser.add_argument('--lr-theta', type=float, default=None, help='Learning rate for symmetry parameter optimization (defaults to lr if not set)')
    parser.add_argument('--objective', type=str, default='virtual_loss', choices=['virtual_loss', 'grad_norm', 'bo_grad_norm', 'virtual_sgd_improve'], help='Teleportation objective (default: virtual_loss)')
    parser.add_argument('--s-param', type=str, default='exp', choices=['exp', 'direct', 'projected'], help='Symmetry parameterization: exp (s=exp(log_s)), direct (s=softplus(u)+eps), or projected (s with projection) (default: exp)')
    parser.add_argument('--teleport', type=int, default=1, help='Enable teleportation (1) or disable for baseline (0) (default: 1)')
    parser.add_argument('--force-nontrivial-s', action='store_true', help='Force nontrivial symmetry scaling for diagnostics')
    parser.add_argument('--theta-target-max-log-s', type=float, default=0.5, help='Target max|log_s| when forcing scaling')
    parser.add_argument('--rescale-lr-post-teleport', action='store_true', help='Scale LR for next SGD step after accepted teleport using ||g||/||g_prime||')
    
    args = parser.parse_args()
    
    # Override with aggressive settings if requested
    if args.aggressive:
        args.lambda_penalty = 1e-6
        args.inner_steps = 50
        if args.lr_theta is None:
            args.lr_theta = 0.1
        print("Using aggressive teleportation settings: lambda=1e-6, inner_steps=50, lr_theta=0.1")
    
    # Parse log_s_clip
    log_s_clip = tuple(map(float, args.log_s_clip.split(',')))
    model_kwargs = model_config_from_budget(args.param_budget)
    param_count = int(sum(p.numel() for p in TinyTransformer(**model_kwargs).parameters()))
    
    # Create output directories
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    runs_dir = outdir / 'runs'
    runs_dir.mkdir(exist_ok=True)
    
    # Store config
    config = {
        'steps': args.steps,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'param_budget': args.param_budget,
        'model_kwargs': model_kwargs,
        'param_count': param_count,
        'teleport_every': args.teleport_every if args.teleport else 0,
        'lambda_penalty': args.lambda_penalty,
        'inner_steps': args.inner_steps,
        'log_s_clip': log_s_clip,
        'objective': args.objective,
        's_param': args.s_param,
        'teleport_enabled': bool(args.teleport),
        'force_nontrivial_s': bool(args.force_nontrivial_s),
        'theta_target_max_log_s': float(args.theta_target_max_log_s),
        'rescale_lr_post_teleport': bool(args.rescale_lr_post_teleport),
    }
    
    print("=" * 60)
    print("Tiny Transformer Benchmark: SGD vs SGD+Teleportation")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"Steps per seed: {args.steps}")
    print(f"Param budget: {args.param_budget}")
    print(f"Model kwargs: {model_kwargs}")
    print(f"Total parameters: {param_count}")
    print(f"Config: {config}")
    print()
    
    # Run all seeds
    all_results = {}
    for seed_idx, seed in enumerate(args.seeds):
        print(f"[{seed_idx+1}/{len(args.seeds)}] Running seed {seed}...", end=' ', flush=True)
        
        (
            baseline_losses,
            teleport_losses,
            active_events,
            init_param,
            first_batch,
            teleport_attempts,
            baseline_grad_norm_all,
            baseline_grad_norm_ffn,
            teleport_grad_norm_all,
            teleport_grad_norm_ffn,
            data_hash,
            param_hash,
        ) = run_single_seed(
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            teleport_every=args.teleport_every if args.teleport else 0,
            lambda_penalty=args.lambda_penalty,
            inner_steps=args.inner_steps,
            log_s_clip=log_s_clip,
            lr_theta=args.lr_theta,
            objective=args.objective,
            s_param=args.s_param,
            force_nontrivial_s=args.force_nontrivial_s,
            theta_target_max_log_s=args.theta_target_max_log_s,
            rescale_lr_post_teleport=args.rescale_lr_post_teleport,
            model_kwargs=model_kwargs,
            device='cpu'
        )
        
        all_results[seed] = {
            'baseline_losses': baseline_losses,
            'teleport_losses': teleport_losses,
            'baseline_grad_norm_all': baseline_grad_norm_all,
            'baseline_grad_norm_ffn': baseline_grad_norm_ffn,
            'teleport_grad_norm_all': teleport_grad_norm_all,
            'teleport_grad_norm_ffn': teleport_grad_norm_ffn,
            'active_events': active_events,
            'init_param_sample': init_param,
            'first_batch_mean': first_batch,
            'teleport_attempts': teleport_attempts,
            'data_hash': data_hash,
            'param_hash': param_hash,
        }
        
        # Save per-seed results
        seed_file = runs_dir / f'seed_{seed}.json'
        with open(seed_file, 'w') as f:
            json.dump(all_results[seed], f, indent=2)
        
        print(f"  Done (active: {active_events}, data_hash={data_hash}, param_hash={param_hash})")
    
    print()
    print("Aggregating results...")
    
    # Aggregate
    summary = aggregate_results(all_results, config)
    
    # Save summary
    summary_file = outdir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_file}")
    
    # Plot
    plot_file = outdir / 'curve.png'
    plot_curves(summary, plot_file)
    # Additional plots with teleport markers
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "plot_grad_norms.py"), "--outdir", str(outdir)],
        check=True
    )
    
    # Print metrics
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Seed diversity check (hash-based)
    diversity = summary['seed_diversity']
    n_unique_data = diversity['unique_data_hashes']
    n_unique_param = diversity['unique_param_hashes']

    print(f"Seed Diversity Check (hash-based):")
    print(f"  Unique data hashes:  {n_unique_data}/{len(summary['seeds'])}")
    print(f"  Unique param hashes: {n_unique_param}/{len(summary['seeds'])}")
    if n_unique_data == 1:
        print("  WARNING: All seeds produced IDENTICAL datasets!")
    if n_unique_param == 1:
        print("  WARNING: All seeds produced IDENTICAL model initializations!")
    if n_unique_data > 1 and n_unique_param > 1:
        print("  OK: Seeds are diverse")
    print()
    
    metrics = summary['metrics']
    print(f"Final Loss (mean ± std):")
    print(f"  Baseline:  {metrics['final_loss_mean']['baseline']:.6f} ± {metrics['final_loss_mean']['baseline_std']:.6f}")
    print(f"  Teleport:  {metrics['final_loss_mean']['teleport']:.6f} ± {metrics['final_loss_mean']['teleport_std']:.6f}")
    print()
    
    if metrics['steps_to_threshold_A']['baseline_mean'] is not None:
        print(f"Steps to Threshold A ({metrics['threshold_A']:.6f}):")
        print(f"  Baseline:  {metrics['steps_to_threshold_A']['baseline_mean']:.1f} ± {metrics['steps_to_threshold_A']['baseline_std']:.1f}")
        print(f"  Teleport:  {metrics['steps_to_threshold_A']['teleport_mean']:.1f} ± {metrics['steps_to_threshold_A']['teleport_std']:.1f}")
        print(f"  Speedup:   {metrics['speedup_A_percent']:.2f}%")
    else:
        print("Steps to Threshold A: N/A (threshold not reached)")
    print()
    
    if metrics['steps_to_threshold_B']['baseline_mean'] is not None:
        print(f"Steps to Threshold B ({metrics['threshold_B']:.6f}):")
        print(f"  Baseline:  {metrics['steps_to_threshold_B']['baseline_mean']:.1f} ± {metrics['steps_to_threshold_B']['baseline_std']:.1f}")
        print(f"  Teleport:  {metrics['steps_to_threshold_B']['teleport_mean']:.1f} ± {metrics['steps_to_threshold_B']['teleport_std']:.1f}")
        print(f"  Speedup:   {metrics['speedup_B_percent']:.2f}%")
    else:
        print("Steps to Threshold B: N/A (threshold not reached)")
    print()

    loss_at_k = metrics.get('loss_at_step_k', {})
    if loss_at_k:
        print("Loss@Step-k:")
        for k in sorted(loss_at_k.keys(), key=lambda x: int(x)):
            row = loss_at_k[k]
            print(
                f"  k={k}: baseline={row['baseline_mean']:.6f} ± {row['baseline_std']:.6f}, "
                f"teleport={row['teleport_mean']:.6f} ± {row['teleport_std']:.6f}"
            )
        print()

    auc = metrics.get('area_under_curve', {})
    if auc:
        print("Area Under Curve (AUC):")
        print(f"  Baseline: {auc['baseline_mean']:.10f} ± {auc['baseline_std']:.10f}")
        print(f"  Teleport: {auc['teleport_mean']:.10f} ± {auc['teleport_std']:.10f}")
        print(f"  Paired ΔAUC (teleport-baseline): {auc['paired_delta_mean']:.10f} ± {auc['paired_delta_std']:.10f}")
        print()

    fixed_speedup = metrics.get('speedup_report_fixed_thresholds', {})
    if fixed_speedup:
        print("Fixed Threshold Speedup Report:")
        for thr in sorted(fixed_speedup.keys(), key=float):
            row = fixed_speedup[thr]
            print(
                f"  threshold={thr}: baseline={row['baseline_mean']:.1f} ± {row['baseline_std']:.1f}, "
                f"teleport={row['teleport_mean']:.1f} ± {row['teleport_std']:.1f}, "
                f"speedup={row['speedup_percent']:.2f}%"
            )
        print()
    
    # Print teleport statistics
    tstats = summary['teleport_stats']
    print(f"Teleport Statistics:")
    print(f"  Total attempts: {tstats['total_attempts']}")
    print(f"  Accepted (ΔJ < threshold): {tstats['accepted_count']} ({tstats['acceptance_rate']:.2f}%)")
    print(f"  Nontrivial (max|log_s| ≥ threshold): {tstats['nontrivial_count']} ({tstats['nontrivial_rate']:.2f}%)")
    print(f"  Active (all criteria met): {tstats['active_count']} ({tstats['active_rate_strict']:.2f}%)")
    
    if tstats['accepted_count'] > 0:
        print(f"\n  Statistics on accepted attempts:")
        print(f"    ΔJ: {tstats['accepted_delta_J_mean']:.3e} ± {tstats['accepted_delta_J_std']:.3e} (min: {tstats['accepted_delta_J_min']:.3e})")
        print(f"    max|log_s|: {tstats['accepted_max_abs_log_s_mean']:.3e} ± {tstats['accepted_max_abs_log_s_std']:.3e}")
        print(f"    mean|log_s|: {tstats['accepted_mean_abs_log_s_mean']:.3e}")
    if tstats['total_attempts'] > 0:
        print(f"\n  Symmetry Magnitude (all attempts):")
        print(f"    max|log_s| mean: {tstats['max_abs_log_s_mean']:.3e}")
        print(f"    max|log_s| median: {tstats['max_abs_log_s_median']:.3e}")
        print(f"    mean|log_s| mean: {tstats['mean_abs_log_s_mean']:.3e}")
        print(f"    s_min mean: {tstats['s_min_mean']:.3e}")
        print(f"    s_max mean: {tstats['s_max_mean']:.3e}")
        print(f"    max|s-1| mean: {tstats['max_abs_s_minus_one_mean']:.3e}")
        print(f"    ΔJ mean (all): {tstats['delta_J_mean_all']:.3e}")
        print(f"    ΔJ median (all): {tstats['delta_J_median_all']:.3e}")
    
    # Print before/after teleport invariance check (Bo's request)
    if tstats['total_attempts'] > 0:
        print(f"\n  Loss Invariance Check (before → after teleport, NO SGD step between):")
        print(f"    ΔL (mean): {tstats['delta_loss_mean']:.3e} ± {tstats['delta_loss_std']:.3e}")
        print(f"    |ΔL| (max): {tstats['delta_loss_max_abs']:.3e}")
        if tstats['delta_loss_max_abs'] > 1e-6:
            print(f"    ⚠️  WARNING: Loss changed significantly (max |ΔL| > 1e-6)")
        else:
            print(f"    ✓ Loss approximately invariant (max |ΔL| ≤ 1e-6)")
        
        print(f"\n  Gradient Norm Change (before → after teleport):")
        print(f"    Δ||∇L|| (mean): {tstats['delta_grad_norm_mean']:.3e} ± {tstats['delta_grad_norm_std']:.3e}")
        print(f"    Δ||∇L|| (median): {tstats['delta_grad_norm_median']:.3e}")
        print(f"    ||∇L|| before: {tstats['grad_norm_before_mean']:.3e}")
        print(f"    ||∇L|| after: {tstats['grad_norm_after_mean']:.3e}")
        print(f"    Δ||∇L||_all (median): {tstats['delta_grad_norm_all_median']:.3e}")
        print(f"    ratio Δ||∇L||_all (median): {tstats['ratio_grad_norm_all_median']:.3e}")
        print(f"    Δ||∇L||_ffn (median): {tstats['delta_grad_norm_ffn_median']:.3e}")
        print(f"    ratio Δ||∇L||_ffn (median): {tstats['ratio_grad_norm_ffn_median']:.3e}")
        print(f"    ||∇L||_all before (median): {tstats['grad_norm_all_before_median']:.3e}")
        print(f"    ||∇L||_ffn before (median): {tstats['grad_norm_ffn_before_median']:.3e}")
        if tstats['delta_grad_norm_mean'] > 0:
            print(f"    ✓ Gradient norm increased on average")
        else:
            print(f"    ✗ Gradient norm decreased on average")
    print()
    print("=" * 60)
    print(f"Artifacts saved to: {outdir.resolve()}")
    print("  - curve.png")
    print("  - summary.json")
    print("  - runs/seed_*.json")
    print("=" * 60)
    
    # One-screen summary for Bo
    if tstats['total_attempts'] > 0:
        print()
        print("=" * 60)
        print("BO'S QUESTIONS - ONE-SCREEN SUMMARY")
        print("=" * 60)
        print(f"max |ΔL|:              {tstats['delta_loss_max_abs']:.3e}")
        print(f"median Δ||∇L||:        {tstats['delta_grad_norm_median']:.3e}")
        print(f"median Δ||∇L||_all:    {tstats['delta_grad_norm_all_median']:.3e}")
        print(f"median ratio_all:      {tstats['ratio_grad_norm_all_median']:.3e}")
        print(f"median Δ||∇L||_ffn:    {tstats['delta_grad_norm_ffn_median']:.3e}")
        print(f"median ratio_ffn:      {tstats['ratio_grad_norm_ffn_median']:.3e}")
        print(f"max|log_s| mean:       {tstats['max_abs_log_s_mean']:.3e}")
        print(f"max|log_s| median:     {tstats['max_abs_log_s_median']:.3e}")
        print(f"attempt_count:         {tstats['attempt_count']}")
        print(f"accepted_count:        {tstats['accepted_count']}")
        print(f"accepted_rate:         {tstats['accepted_rate']:.2f}%")
        print(f"active_count_strict:   {tstats['active_count_strict']}")
        print(f"active_rate_strict:    {tstats['active_rate_strict']:.2f}%")
        print("=" * 60)

    print()
    print("Recommended benchmark commands:")
    print("  Natural teleport (headline):")
    print(
        "    python3 scripts/bench_tiny_transformer.py "
        "--teleport=1 --objective=virtual_sgd_improve --s-param=projected "
        "--teleport-every 20 --lambda-penalty 0 --inner-steps 100 --lr-theta 0.1 "
        "--log-s-clip=-2,2 "
        f"--param-budget={args.param_budget} --steps {args.steps} --lr {args.lr} --batch-size {args.batch_size} "
        f"--seeds {' '.join(map(str, args.seeds))} "
        "--outdir results/recommended_natural"
    )
    print("  Forced teleport (diagnostic only):")
    print(
        "    python3 scripts/bench_tiny_transformer.py "
        "--teleport=1 --objective=virtual_sgd_improve --s-param=projected "
        "--teleport-every 20 --lambda-penalty 0 --inner-steps 100 --lr-theta 0.1 "
        "--log-s-clip=-2,2 --force-nontrivial-s --theta-target-max-log-s 0.2 "
        f"--param-budget={args.param_budget} --steps {args.steps} --lr {args.lr} --batch-size {args.batch_size} "
        f"--seeds {' '.join(map(str, args.seeds))} "
        "--outdir results/recommended_forced_diag"
    )


if __name__ == '__main__':
    main()

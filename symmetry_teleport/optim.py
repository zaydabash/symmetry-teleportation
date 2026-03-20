"""
TeleportSGD: PyTorch optimizer with symmetry teleportation.

This module provides a drop-in replacement for torch.optim.SGD that adds
optional teleportation steps based on symmetry transformations.
"""

import torch
import math
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np

from .teleport import teleport_ffn_diagonal
from .groups import ScalarRescalingGroup


class TeleportSGD(Optimizer):
    """
    SGD optimizer with optional symmetry teleportation.
    
    This optimizer combines standard SGD with periodic teleportation steps that
    exploit loss-invariant symmetries to potentially improve convergence.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
        teleport_every: Apply teleportation every N steps (0 to disable) (default: 0)
        teleport_config: Dict with teleportation settings:
            - model: Model instance (required if teleport_every > 0)
            - layer_idx: Index of layer to teleport (default: 0)
            - X_teleport: Input data for teleportation (required)
            - Y_teleport: Target data for teleportation (required)
            - loss_fn: Loss function (required)
            - lr_theta: Learning rate for symmetry parameter optimization (default: 1e-2)
            - inner_steps: Number of optimization steps (default: 20)
            - lambda_penalty: Penalty weight for regularization (default: 1e-3)
            - acceptance_threshold: ΔJ threshold for acceptance (default: -1e-9)
            - min_log_s_magnitude: Minimum |log_s| for nontrivial (default: 1e-6)
            - log_s_clip: Clamping range for log_s (default: (-2.0, 2.0))
            - objective: 'virtual_loss', 'grad_norm', or 'virtual_sgd_improve' (default: 'virtual_loss')
            - s_param: 'exp', 'direct', or 'projected' (default: 'exp')
            - force_nontrivial_s: Force nontrivial scaling (diagnostic, default: False)
            - theta_target_max_log_s: Target max|log_s| for forced scaling (default: 0.5)
            - rescale_lr_post_teleport: If True, scale LR for next SGD step by ||g||/||g'|| (default: False)
    
    Example:
        >>> model = TinyTransformer()
        >>> optimizer = TeleportSGD(
        ...     model.parameters(),
        ...     lr=0.01,
        ...     teleport_every=5,
        ...     teleport_config={
        ...         'model': model,
        ...         'layer_idx': 0,
        ...         'X_teleport': X_batch,
        ...         'Y_teleport': Y_batch,
        ...         'loss_fn': nn.MSELoss()
        ...     }
        ... )
        >>> 
        >>> # Training loop
        >>> for X, Y in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(X), Y)
        ...     loss.backward()
        ...     optimizer.step()  # Automatically applies teleportation when scheduled
    """
    
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, teleport_every=0, teleport_config=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        super(TeleportSGD, self).__init__(params, defaults)
        
        # Teleportation settings
        self.teleport_every = teleport_every
        self.step_count = 0
        self.teleport_config = teleport_config or {}
        
        # Teleportation hyperparameters
        self.model = self.teleport_config.get('model', None)
        self.layer_idx = self.teleport_config.get('layer_idx', 0)
        self.X_teleport = self.teleport_config.get('X_teleport', None)
        self.Y_teleport = self.teleport_config.get('Y_teleport', None)
        self.loss_fn = self.teleport_config.get('loss_fn', None)
        self.lr_theta = self.teleport_config.get('lr_theta', 1e-2)
        self.inner_steps = self.teleport_config.get('inner_steps', 20)
        self.lambda_penalty = self.teleport_config.get('lambda_penalty', 1e-3)
        self.acceptance_threshold = self.teleport_config.get('acceptance_threshold', -1e-9)
        self.min_log_s_magnitude = self.teleport_config.get('min_log_s_magnitude', 1e-6)
        self.log_s_clip = self.teleport_config.get('log_s_clip', (-2.0, 2.0))
        self.objective = self.teleport_config.get('objective', 'virtual_loss')
        self.s_param = self.teleport_config.get('s_param', 'exp')
        self.force_nontrivial_s = self.teleport_config.get('force_nontrivial_s', False)
        self.theta_target_max_log_s = self.teleport_config.get('theta_target_max_log_s', 0.5)
        self.rescale_lr_post_teleport = self.teleport_config.get('rescale_lr_post_teleport', False)
        self.debug_first_attempt = self.teleport_config.get('debug_first_attempt', False)
        self._debug_attempt_done = False
        self._next_step_lr_scale = None
        
        # Teleportation statistics
        self.teleport_attempts = []
        self.teleport_active_steps = []
        
        # Validate teleportation config if enabled
        if self.teleport_every > 0:
            if self.model is None:
                raise ValueError("teleport_config['model'] is required when teleport_every > 0")
            if self.X_teleport is None or self.Y_teleport is None:
                raise ValueError("teleport_config['X_teleport'] and 'Y_teleport' are required")
            if self.loss_fn is None:
                raise ValueError("teleport_config['loss_fn'] is required")
    
    def __setstate__(self, state):
        super(TeleportSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        
        Also applies teleportation if scheduled (every teleport_every steps).
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            Loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Standard SGD step (with no_grad)
        lr_scale_this_step = self._next_step_lr_scale
        with torch.no_grad():
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                effective_lr = group['lr']
                if lr_scale_this_step is not None:
                    effective_lr = group['lr'] * lr_scale_this_step
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    
                    p.add_(d_p, alpha=-effective_lr)
        if lr_scale_this_step is not None:
            # Apply post-teleport LR rescaling for exactly one SGD step.
            self._next_step_lr_scale = None
        
        self.step_count += 1
        
        # Apply teleportation if scheduled (needs gradients enabled)
        if self.teleport_every > 0 and self.step_count % self.teleport_every == 0:
            self._apply_teleportation()
        
        return loss
    
    def _apply_teleportation(self):
        """
        Apply teleportation step using diagonal scaling symmetry.
        
        This method is called automatically by step() when scheduled.
        """
        if self.X_teleport is None or self.Y_teleport is None or self.loss_fn is None:
            return
        
        # Get current learning rate (use first param group)
        lr = self.param_groups[0]['lr']
        
        # Save parameter state before teleportation
        params_before = {}
        for name, p in self.model.named_parameters():
            params_before[name] = p.detach().clone()
        
        # Compute parameter norm before
        norm_before = self._flat_param_norm(self.model)
        
        # Run teleportation search
        search_objective = self.objective
        debug_inner_log = self.debug_first_attempt and not self._debug_attempt_done
        s_best, J_before_val, J_best_val, diagnostics = teleport_ffn_diagonal(
            self.model,
            self.layer_idx,
            self.X_teleport,
            self.loss_fn,
            self.Y_teleport,
            lr_theta=self.lr_theta,
            steps=self.inner_steps,
            log_s_clip=self.log_s_clip,
            lr=lr,
            lambda_penalty=self.lambda_penalty,
            objective=search_objective,
            s_param=self.s_param,
            force_nontrivial_s=self.force_nontrivial_s,
            theta_target_max_log_s=self.theta_target_max_log_s,
            debug_inner_log=debug_inner_log
        )
        if debug_inner_log:
            self._debug_attempt_done = True
        
        # Compute delta J
        delta_J = J_best_val - J_before_val
        
        # Compute log_s statistics
        log_s_best = torch.log(s_best.clamp_min(1e-12))
        max_abs_log_s = diagnostics.get('max_abs_log_s', float(log_s_best.abs().max().item()))
        mean_abs_log_s = diagnostics.get('mean_abs_log_s', float(log_s_best.abs().mean().item()))
        log_s_norm = float(log_s_best.norm().item())
        s_min = diagnostics.get('s_min', float(s_best.min().item()))
        s_max = diagnostics.get('s_max', float(s_best.max().item()))
        max_abs_s_minus_one = diagnostics.get('max_abs_s_minus_one', float((s_best - 1.0).abs().max().item()))

        # Rejection guards
        finite_vals = [
            diagnostics.get('loss_before', 0.0),
            diagnostics.get('loss_after', 0.0),
            diagnostics.get('grad_norm_before', 0.0),
            diagnostics.get('grad_norm_after', 0.0),
        ]
        finite_ok = all(math.isfinite(float(v)) for v in finite_vals)
        log_s_hi = float(self.log_s_clip[1])
        max_s_bound = math.exp(log_s_hi)
        bounds_ok = (s_max <= max_s_bound) and (max_abs_log_s <= log_s_hi)
        
        # Determine acceptance criteria
        epsilon = 1e-10
        L_baseline_virtual = None
        L_tp_virtual = None
        delta_virtual = None
        grad_norm_virtual_before = None
        grad_norm_virtual_after = None
        if self.objective == 'virtual_sgd_improve':
            L_baseline_virtual, L_tp_virtual = self._compute_virtual_sgd_losses_stateful(s_best, lr)
            delta_virtual = L_tp_virtual - L_baseline_virtual
            grad_norm_virtual_before = getattr(self, '_last_virtual_grad_norm_before', None)
            grad_norm_virtual_after = getattr(self, '_last_virtual_grad_norm_after', None)
            accepted = (L_tp_virtual < (L_baseline_virtual - epsilon)) and finite_ok and bounds_ok
        else:
            accepted = (delta_J < self.acceptance_threshold) and finite_ok and bounds_ok
        nontrivial = max_abs_log_s >= self.min_log_s_magnitude
        
        # Apply transformation if accepted
        if accepted:
            linear1, linear2 = self.model.get_ffn_layers(self.layer_idx)
            ScalarRescalingGroup.apply_transform(linear1, linear2, s_best)
        
        # Optional one-step post-teleport LR rescaling
        lr_scale_next = None
        if accepted and self.rescale_lr_post_teleport and self.objective == 'virtual_sgd_improve':
            denom = grad_norm_virtual_after + 1e-12
            lr_scale_next = grad_norm_virtual_before / denom
            if math.isfinite(lr_scale_next) and lr_scale_next > 0.0:
                self._next_step_lr_scale = lr_scale_next
            else:
                self._next_step_lr_scale = None
                lr_scale_next = None

        # Compute parameter change
        max_w_delta = self._max_param_delta(self.model, params_before)
        changed = max_w_delta >= 1e-6
        
        # Strict activation: all three criteria must be true
        teleport_active_strict = accepted and nontrivial and changed
        
        # Compute parameter norm after
        norm_after = self._flat_param_norm(self.model)
        
        # Compute invariance check (output change on fixed batch)
        max_inv_delta = self._check_invariance(params_before)
        
        # Log attempt (including before/after metrics)
        attempt_log = {
            'step_count': self.step_count,
            'step': self.step_count,
            'accepted': accepted,
            'nontrivial': nontrivial,
            'changed': changed,
            'active_strict': teleport_active_strict,
            'finite_ok': finite_ok,
            'bounds_ok': bounds_ok,
            'objective': self.objective,
            'J_before_val': J_before_val,
            'J_best_val': J_best_val,
            'delta_J': delta_J,
            'L_baseline_virtual': L_baseline_virtual,
            'L_tp_virtual': L_tp_virtual,
            'delta_virtual': delta_virtual,
            'grad_norm_virtual_before': grad_norm_virtual_before,
            'grad_norm_virtual_after': grad_norm_virtual_after,
            'lr_scale_next_step': lr_scale_next,
            'max_abs_log_s': max_abs_log_s,
            'mean_abs_log_s': mean_abs_log_s,
            'log_s_norm': log_s_norm,
            's_min': s_min,
            's_max': s_max,
            'max_abs_s_minus_one': max_abs_s_minus_one,
            'force_nontrivial_applied': diagnostics.get('force_nontrivial_applied', False),
            'max_abs_log_s_before_force': diagnostics.get('max_abs_log_s_before_force', max_abs_log_s),
            'force_target_max_log_s': diagnostics.get('force_target_max_log_s', self.theta_target_max_log_s),
            'max_w_delta': max_w_delta,
            'norm_before': norm_before,
            'norm_after': norm_after,
            'initial_grad_norm': diagnostics.get('initial_grad_norm', 0.0),
            'max_inv_delta': max_inv_delta,
            # Before/after teleport metrics (Bo's request)
            'loss_before': diagnostics.get('loss_before', 0.0),
            'loss_after': diagnostics.get('loss_after', 0.0),
            'grad_norm_all_before': diagnostics.get('grad_norm_all_before', 0.0),
            'grad_norm_all_after': diagnostics.get('grad_norm_all_after', 0.0),
            'grad_norm_ffn_before': diagnostics.get('grad_norm_ffn_before', 0.0),
            'grad_norm_ffn_after': diagnostics.get('grad_norm_ffn_after', 0.0),
            'loss_before_teleport': diagnostics.get('loss_before', 0.0),
            'loss_after_teleport': diagnostics.get('loss_after', 0.0),
            'delta_loss_teleport': diagnostics.get('delta_loss', 0.0),
            'grad_norm_before_teleport': diagnostics.get('grad_norm_before', 0.0),
            'grad_norm_after_teleport': diagnostics.get('grad_norm_after', 0.0),
            'delta_grad_norm_teleport': diagnostics.get('delta_grad_norm', 0.0),
            'grad_norm_all_before_teleport': diagnostics.get('grad_norm_all_before', 0.0),
            'grad_norm_all_after_teleport': diagnostics.get('grad_norm_all_after', 0.0),
            'delta_grad_norm_all_teleport': diagnostics.get('delta_grad_norm_all', 0.0),
            'ratio_grad_norm_all_teleport': diagnostics.get('ratio_grad_norm_all', 0.0),
            'grad_norm_ffn_before_teleport': diagnostics.get('grad_norm_ffn_before', 0.0),
            'grad_norm_ffn_after_teleport': diagnostics.get('grad_norm_ffn_after', 0.0),
            'delta_grad_norm_ffn_teleport': diagnostics.get('delta_grad_norm_ffn', 0.0),
            'ratio_grad_norm_ffn_teleport': diagnostics.get('ratio_grad_norm_ffn', 0.0)
        }
        
        self.teleport_attempts.append(attempt_log)
        
        if teleport_active_strict:
            self.teleport_active_steps.append(self.step_count)
        
        # If not accepted, restore parameters
        if not accepted:
            for name, p in self.model.named_parameters():
                p.data.copy_(params_before[name])

    def _clone_state_dict(self):
        """Clone model state_dict tensors."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _load_state_dict(self, sd):
        """Restore model state_dict exactly."""
        self.model.load_state_dict(sd, strict=True)

    def _compute_loss_and_grads(self):
        """
        Compute loss and gradients on teleport batch.

        Returns:
            (loss_value, grads_list_aligned_with_model_parameters)
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(self.X_teleport)
        loss = self.loss_fn(out, self.Y_teleport)
        loss.backward()
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                grads.append(None)
            else:
                grads.append(p.grad.detach().clone())
        return float(loss.item()), grads

    def _apply_virtual_sgd_step(self, grads, lr):
        """Apply one in-place virtual SGD step to current model parameters."""
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), grads):
                if g is None:
                    continue
                p.add_(g, alpha=-lr)

    @staticmethod
    def _grad_norm_from_grads(grads):
        """Compute L2 norm from a list of gradients (entries may be None)."""
        total = 0.0
        for g in grads:
            if g is None:
                continue
            total += float((g.detach() * g.detach()).sum().item())
        return total ** 0.5

    def _compute_virtual_sgd_losses_stateful(self, s_candidate, lr):
        """
        Compute virtual k-step loss comparison without parameter-name lookups.

        Uses k=5 virtual SGD steps at lr_virtual = 2*lr, matching the inner
        objective_virtual_sgd_improve function for consistent search/accept semantics.

        Returns:
            (L_baseline_virtual, L_tp_virtual)
        """
        k_virtual = 5
        lr_virtual = 2.0 * lr

        sd0 = self._clone_state_dict()
        was_training = self.model.training
        self.model.eval()
        try:
            # BASELINE: k virtual SGD steps from theta (no teleport)
            self._load_state_dict(sd0)
            _, grads0 = self._compute_loss_and_grads()
            norm_before = self._grad_norm_from_grads(grads0)
            self._apply_virtual_sgd_step(grads0, lr_virtual)
            for _ in range(k_virtual - 1):
                _, grads_step = self._compute_loss_and_grads()
                self._apply_virtual_sgd_step(grads_step, lr_virtual)
            with torch.no_grad():
                L_baseline_virtual = float(self.loss_fn(self.model(self.X_teleport), self.Y_teleport).item())

            # TELEPORTED: apply candidate teleport, then k virtual SGD steps
            self._load_state_dict(sd0)
            linear1, linear2 = self.model.get_ffn_layers(self.layer_idx)
            ScalarRescalingGroup.apply_transform(linear1, linear2, s_candidate)
            _, grads1 = self._compute_loss_and_grads()
            norm_after = self._grad_norm_from_grads(grads1)
            self._apply_virtual_sgd_step(grads1, lr_virtual)
            for _ in range(k_virtual - 1):
                _, grads_step = self._compute_loss_and_grads()
                self._apply_virtual_sgd_step(grads_step, lr_virtual)
            with torch.no_grad():
                L_tp_virtual = float(self.loss_fn(self.model(self.X_teleport), self.Y_teleport).item())
        finally:
            # Restore original parameters and mode; never leave side effects.
            self._load_state_dict(sd0)
            if was_training:
                self.model.train()
            self.model.zero_grad(set_to_none=True)

        # Cache grad norms at teleport point (before virtual steps) for optional LR rescaling.
        self._last_virtual_grad_norm_before = norm_before
        self._last_virtual_grad_norm_after = norm_after
        return L_baseline_virtual, L_tp_virtual
    
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
            for name, p in model.named_parameters():
                if name in params_before:
                    mx = max(mx, (p.detach() - params_before[name]).abs().max().item())
            return mx
    
    def _check_invariance(self, params_before):
        """
        Check function invariance by comparing outputs before/after transformation.
        
        Returns:
            max_inv_delta: Maximum absolute difference in outputs
        """
        if self.X_teleport is None:
            return 0.0
        
        with torch.no_grad():
            # Output after transformation
            out_after = self.model(self.X_teleport)
            
            # Temporarily restore old parameters
            params_current = {}
            for name, p in self.model.named_parameters():
                params_current[name] = p.detach().clone()
                p.data.copy_(params_before[name])
            
            # Output before transformation
            out_before = self.model(self.X_teleport)
            
            # Restore current parameters
            for name, p in self.model.named_parameters():
                p.data.copy_(params_current[name])
            
            # Compute max difference
            max_inv_delta = float((out_after - out_before).abs().max().item())
            
        return max_inv_delta
    
    def teleport_stats(self):
        """
        Get teleportation statistics.
        
        Returns:
            Dict with counts and recent metrics:
                - total_attempts: Total number of teleportation attempts
                - accepted_count: Number of accepted teleportations
                - nontrivial_count: Number of nontrivial teleportations
                - active_count: Number of strictly active teleportations
                - acceptance_rate: Percentage of accepted attempts
                - nontrivial_rate: Percentage of nontrivial attempts
                - active_rate_strict: Percentage of strictly active attempts
                - recent_delta_J: List of recent ΔJ values (last 10)
                - recent_max_log_s: List of recent max|log_s| values (last 10)
        """
        if not self.teleport_attempts:
            return {
                'total_attempts': 0,
                'accepted_count': 0,
                'nontrivial_count': 0,
                'active_count': 0,
                'acceptance_rate': 0.0,
                'nontrivial_rate': 0.0,
                'active_rate_strict': 0.0,
                'recent_delta_J': [],
                'recent_max_log_s': []
            }
        
        total = len(self.teleport_attempts)
        accepted = sum(1 for a in self.teleport_attempts if a['accepted'])
        nontrivial = sum(1 for a in self.teleport_attempts if a['nontrivial'])
        active = sum(1 for a in self.teleport_attempts if a['active_strict'])
        
        recent_delta_J = [a['delta_J'] for a in self.teleport_attempts[-10:]]
        recent_max_log_s = [a['max_abs_log_s'] for a in self.teleport_attempts[-10:]]
        
        return {
            'total_attempts': total,
            'accepted_count': accepted,
            'nontrivial_count': nontrivial,
            'active_count': active,
            'acceptance_rate': 100.0 * accepted / total if total > 0 else 0.0,
            'nontrivial_rate': 100.0 * nontrivial / total if total > 0 else 0.0,
            'active_rate_strict': 100.0 * active / total if total > 0 else 0.0,
            'recent_delta_J': recent_delta_J,
            'recent_max_log_s': recent_max_log_s
        }

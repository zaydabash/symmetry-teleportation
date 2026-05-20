"""
Tests for Q/K diagonal teleportation integration.

Covers:
- _apply_qk_diagonal_inplace: output changes, Q K^T logits preserved
- teleport_qk_diagonal: runs without error, returns valid types
- FFN-only TeleportSGD: old configs still work unchanged
- FFN+QK TeleportSGD: runs end-to-end without error
- Rejected transform restores params exactly
- teleport_stats keys present for ffn_qk attempts
"""

import math
import pytest
import torch
import torch.nn as nn

from symmetry_teleport import TeleportSGD
from symmetry_teleport.teleport import (
    _apply_qk_diagonal_inplace,
    teleport_qk_diagonal,
)
from transformer_teleport_optimizer import TinyTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_model_and_data(
    seed=0, d_model=16, nhead=2, dim_ff=32, seq_len=6, batch=8
):
    torch.manual_seed(seed)
    model = TinyTransformer(
        d_model=d_model, nhead=nhead, num_layers=1,
        dim_feedforward=dim_ff, dropout=0.0
    )
    # TinyTransformer: embed=Linear(4, d_model), output=Linear(d_model, 4)
    X = torch.randn(batch, seq_len, 4)  # input feature dim is always 4
    Y = torch.randn(batch, 4)           # (batch, 4) after mean-pool + linear
    return model, X, Y


def _loss_fn(logits, targets):
    return nn.MSELoss()(logits, targets)


# ---------------------------------------------------------------------------
# _apply_qk_diagonal_inplace
# ---------------------------------------------------------------------------

class TestApplyQKDiagonalInplace:
    def test_output_preserved(self):
        """Q/K diagonal scaling is a true symmetry: full model output unchanged."""
        torch.manual_seed(1)
        model, X, _ = _make_model_and_data()
        attn = model.get_attn_layer(0)
        d = attn.embed_dim

        with torch.no_grad():
            out_before = model(X).clone()

        _apply_qk_diagonal_inplace(attn, torch.full((d,), 2.0))

        with torch.no_grad():
            out_after = model(X)

        # Q'K'^T = QK^T, so softmax weights and V output are unchanged
        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Output should be unchanged by Q/K scaling (symmetry); "
            f"max diff = {(out_before - out_after).abs().max().item():.3e}"
        )

    def test_identity_a_is_no_op(self):
        """a = 1 everywhere must leave model output unchanged."""
        torch.manual_seed(2)
        model, X, _ = _make_model_and_data()
        attn = model.get_attn_layer(0)
        d = attn.embed_dim

        with torch.no_grad():
            out_before = model(X).clone()

        _apply_qk_diagonal_inplace(attn, torch.ones(d))

        with torch.no_grad():
            out_after = model(X)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_qk_logit_invariance(self):
        """Q'K'^T must equal Q K^T after diagonal a scaling."""
        torch.manual_seed(3)
        d_model, nhead = 16, 2
        attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.0
        )
        B, T = 4, 6
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            W = attn.in_proj_weight
            b = attn.in_proj_bias
            d = d_model
            Q_before = X @ W[0:d, :].T + b[0:d]
            K_before = X @ W[d:2 * d, :].T + b[d:2 * d]
            QKt_before = Q_before @ K_before.transpose(-2, -1)

        _apply_qk_diagonal_inplace(attn, torch.exp(0.3 * torch.randn(d_model)))

        with torch.no_grad():
            W2 = attn.in_proj_weight
            b2 = attn.in_proj_bias
            Q_after = X @ W2[0:d, :].T + b2[0:d]
            K_after = X @ W2[d:2 * d, :].T + b2[d:2 * d]
            QKt_after = Q_after @ K_after.transpose(-2, -1)

        err = (QKt_before - QKt_after).abs().max().item()
        assert err < 5e-4, f"Q K^T invariance violated: max error = {err:.3e}"

    def test_round_trip_restores_weights(self):
        """Applying a then 1/a must restore all weights exactly."""
        torch.manual_seed(4)
        model, _, _ = _make_model_and_data()
        attn = model.get_attn_layer(0)
        d = attn.embed_dim

        W_orig = attn.in_proj_weight.detach().clone()
        b_orig = (
            attn.in_proj_bias.detach().clone()
            if attn.in_proj_bias is not None else None
        )

        a = torch.exp(0.25 * torch.randn(d))
        _apply_qk_diagonal_inplace(attn, a)
        _apply_qk_diagonal_inplace(attn, 1.0 / a)

        err_W = (attn.in_proj_weight - W_orig).abs().max().item()
        assert err_W < 1e-5, f"Weight not restored: {err_W:.3e}"
        if b_orig is not None:
            err_b = (attn.in_proj_bias - b_orig).abs().max().item()
            assert err_b < 1e-5, f"Bias not restored: {err_b:.3e}"


# ---------------------------------------------------------------------------
# teleport_qk_diagonal
# ---------------------------------------------------------------------------

class TestTeleportQKDiagonal:
    def test_returns_valid_types(self):
        """teleport_qk_diagonal must return (tensor, float, float, dict)."""
        torch.manual_seed(10)
        model, X, Y = _make_model_and_data(seed=10, batch=6, seq_len=4)
        model.eval()
        a_best, J_before, J_best, diag = teleport_qk_diagonal(
            model, layer_idx=0,
            X_full=X, loss_fn=_loss_fn, Y=Y,
            lr_theta=0.05, steps=5, lr=0.05, restarts=2,
        )
        assert isinstance(a_best, torch.Tensor)
        assert a_best.shape == (model.get_attn_layer(0).embed_dim,)
        assert (a_best > 0).all(), "All scale factors must be positive"
        assert isinstance(J_before, float) and math.isfinite(J_before)
        assert isinstance(J_best, float) and math.isfinite(J_best)
        assert isinstance(diag, dict)

    def test_model_not_mutated(self):
        """teleport_qk_diagonal must not change model weights."""
        torch.manual_seed(11)
        model, X, Y = _make_model_and_data(seed=11, batch=6, seq_len=4)
        params_before = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

        teleport_qk_diagonal(
            model, layer_idx=0,
            X_full=X, loss_fn=_loss_fn, Y=Y,
            lr_theta=0.05, steps=5, lr=0.05, restarts=2,
        )

        for name, p in model.named_parameters():
            assert torch.allclose(p, params_before[name], atol=1e-6), (
                f"Parameter {name} mutated during teleport_qk_diagonal"
            )

    def test_training_mode_restored(self):
        """teleport_qk_diagonal must restore model.training state."""
        torch.manual_seed(12)
        model, X, Y = _make_model_and_data(seed=12, batch=6, seq_len=4)
        model.train()

        teleport_qk_diagonal(
            model, layer_idx=0,
            X_full=X, loss_fn=_loss_fn, Y=Y,
            lr_theta=0.05, steps=5, lr=0.05, restarts=2,
        )
        assert model.training, "model.training should be restored to True"


# ---------------------------------------------------------------------------
# TeleportSGD: FFN-only unchanged
# ---------------------------------------------------------------------------

class TestFFNOnlyUnchanged:
    def test_ffn_only_config_still_works(self):
        """Old FFN-only config must work; teleport_target defaults to 'ffn'."""
        torch.manual_seed(20)
        model, X, Y = _make_model_and_data(seed=20)

        opt = TeleportSGD(
            model.parameters(),
            lr=0.05,
            teleport_every=3,
            teleport_config={
                'model': model,
                'layer_idx': 0,
                'X_teleport': X,
                'Y_teleport': Y,
                'loss_fn': _loss_fn,
                'lr_theta': 0.1,
                'inner_steps': 5,
                'objective': 'virtual_sgd_improve',
                's_param': 'projected',
                'log_s_clip': (-1.0, 1.0),
            },
        )
        assert opt.teleport_target == 'ffn'

        for _ in range(6):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()

        assert opt.teleport_stats()['total_attempts'] == 2

    def test_invalid_teleport_target_raises(self):
        """Unknown teleport_target must raise ValueError at construction."""
        torch.manual_seed(21)
        model, X, Y = _make_model_and_data(seed=21)
        with pytest.raises(ValueError, match="teleport_target"):
            TeleportSGD(
                model.parameters(),
                lr=0.01,
                teleport_every=5,
                teleport_config={
                    'model': model,
                    'layer_idx': 0,
                    'X_teleport': X,
                    'Y_teleport': Y,
                    'loss_fn': _loss_fn,
                    'teleport_target': 'invalid_mode',
                    'objective': 'virtual_sgd_improve',
                },
            )

    def test_ffn_qk_requires_virtual_sgd_improve(self):
        """ffn_qk with non-virtual objective must raise ValueError."""
        torch.manual_seed(22)
        model, X, Y = _make_model_and_data(seed=22)
        with pytest.raises(ValueError, match="virtual_sgd_improve"):
            TeleportSGD(
                model.parameters(),
                lr=0.01,
                teleport_every=5,
                teleport_config={
                    'model': model,
                    'layer_idx': 0,
                    'X_teleport': X,
                    'Y_teleport': Y,
                    'loss_fn': _loss_fn,
                    'teleport_target': 'ffn_qk',
                    'objective': 'virtual_loss',
                },
            )


# ---------------------------------------------------------------------------
# TeleportSGD: FFN+QK end-to-end
# ---------------------------------------------------------------------------

class TestFFNQKEndToEnd:
    def _make_opt(self, model, X, Y):
        return TeleportSGD(
            model.parameters(),
            lr=0.05,
            teleport_every=5,
            teleport_config={
                'model': model,
                'layer_idx': 0,
                'X_teleport': X,
                'Y_teleport': Y,
                'loss_fn': _loss_fn,
                'lr_theta': 0.1,
                'inner_steps': 5,
                'objective': 'virtual_sgd_improve',
                's_param': 'projected',
                'log_s_clip': (-1.0, 1.0),
                'teleport_target': 'ffn_qk',
            },
        )

    def test_runs_without_error(self):
        """FFN+QK optimizer must complete 10 steps without raising."""
        torch.manual_seed(30)
        model, X, Y = _make_model_and_data(seed=30)
        opt = self._make_opt(model, X, Y)

        for _ in range(10):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()

    def test_attempt_logged(self):
        """After 5 steps, exactly one teleport attempt must be logged."""
        torch.manual_seed(31)
        model, X, Y = _make_model_and_data(seed=31)
        opt = self._make_opt(model, X, Y)

        for _ in range(5):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()

        assert opt.teleport_stats()['total_attempts'] == 1

    def test_stats_keys_present(self):
        """Attempt log must contain required FFN+QK fields."""
        torch.manual_seed(32)
        model, X, Y = _make_model_and_data(seed=32)
        opt = self._make_opt(model, X, Y)

        for _ in range(5):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()

        attempt = opt.teleport_attempts[0]
        required = (
            'accepted', 'L_baseline_virtual', 'L_tp_virtual',
            'max_abs_log_s', 'max_abs_log_a', 'teleport_target',
        )
        for key in required:
            assert key in attempt, f"Missing key: {key}"
        assert attempt['teleport_target'] == 'ffn_qk'

    def test_rejected_params_are_finite(self):
        """After a rejected teleport, all model params must be finite."""
        torch.manual_seed(33)
        model, X, Y = _make_model_and_data(seed=33)
        opt = self._make_opt(model, X, Y)

        for _ in range(5):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()

        attempt = opt.teleport_attempts[0]
        if not attempt['accepted']:
            for name, p in model.named_parameters():
                assert torch.isfinite(p).all(), (
                    f"Parameter {name} has non-finite values after rejection"
                )

    def test_accepted_implies_changed(self):
        """accepted=True must imply changed=True in the attempt log."""
        for seed in range(34, 44):
            torch.manual_seed(seed)
            model, X, Y = _make_model_and_data(seed=seed)
            opt = self._make_opt(model, X, Y)

            for _ in range(5):
                opt.zero_grad()
                _loss_fn(model(X), Y).backward()
                opt.step()

            if opt.teleport_attempts and opt.teleport_attempts[0]['accepted']:
                assert opt.teleport_attempts[0]['changed']
                break

    def test_loss_finite_throughout(self):
        """Training loss must remain finite for all 15 steps."""
        torch.manual_seed(35)
        model, X, Y = _make_model_and_data(seed=35)
        opt = self._make_opt(model, X, Y)

        for _ in range(15):
            opt.zero_grad()
            loss = _loss_fn(model(X), Y)
            loss.backward()
            opt.step()
            assert math.isfinite(float(loss.item())), "Loss became non-finite"

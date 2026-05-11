"""
Tests for AttentionQKGroup: Q/K projection symmetry for self-attention.

Three test categories:
  A. Raw attention-logit invariance  — max|QK^T - Q'K'^T| near machine precision
  B. Full attention output invariance — forward pass unchanged after transform
  C. Multiple random trials           — invariance holds across seeds and A samples
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symmetry_teleport.groups import AttentionQKGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_invertible(d: int, scale: float = 0.3, seed: int = 0) -> torch.Tensor:
    """Return I + scale*randn(d,d); well-conditioned for small scale."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.eye(d) + scale * torch.randn(d, d, generator=g)


def _raw_logits(W_Q, W_K, b_Q, b_K, X):
    """Compute Q @ K.T given raw weight matrices and input X."""
    Q = X @ W_Q.T + b_Q  # (B, T, d)
    K = X @ W_K.T + b_K  # (B, T, d)
    return Q @ K.transpose(-2, -1)  # (B, T, T)


# ---------------------------------------------------------------------------
# A. Raw attention-logit invariance
# ---------------------------------------------------------------------------

class TestRawLogitInvariance:
    """
    Verify max|QK^T - Q'K'^T| at near machine precision.

    Operates directly on weight matrices — no nn.MultiheadAttention involved.
    Any invertible A works for this test (no multi-head split).
    """

    def test_basic_float32(self):
        torch.manual_seed(42)
        B, T, d_model, d_k = 4, 8, 16, 16
        X = torch.randn(B, T, d_model)
        W_Q = torch.randn(d_k, d_model)
        W_K = torch.randn(d_k, d_model)
        b_Q = torch.randn(d_k)
        b_K = torch.randn(d_k)

        A = _random_invertible(d_k, scale=0.3, seed=1)
        A_inv = torch.linalg.inv(A)

        QKt_before = _raw_logits(W_Q, W_K, b_Q, b_K, X)

        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        b_Q2 = b_Q @ A
        b_K2 = b_K @ A_inv.T
        QKt_after = _raw_logits(W_Q2, W_K2, b_Q2, b_K2, X)

        err = (QKt_before - QKt_after).abs().max().item()
        # float32 bound: ||Q||·cond(A)·ε_mach·d·||K|| ~ 4·50·1.2e-7·16·4 ≈ 1.5e-3
        assert err < 1e-2, f"float32 logit error {err:.3e} exceeds 1e-2"

    def test_basic_float64(self):
        """float64 should be near machine precision (~1e-12)."""
        torch.manual_seed(42)
        B, T, d_model, d_k = 4, 8, 16, 16
        X = torch.randn(B, T, d_model, dtype=torch.float64)
        W_Q = torch.randn(d_k, d_model, dtype=torch.float64)
        W_K = torch.randn(d_k, d_model, dtype=torch.float64)
        b_Q = torch.randn(d_k, dtype=torch.float64)
        b_K = torch.randn(d_k, dtype=torch.float64)

        A = _random_invertible(d_k, scale=0.3, seed=1).double()
        A_inv = torch.linalg.inv(A)

        QKt_before = _raw_logits(W_Q, W_K, b_Q, b_K, X)

        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        b_Q2 = b_Q @ A
        b_K2 = b_K @ A_inv.T
        QKt_after = _raw_logits(W_Q2, W_K2, b_Q2, b_K2, X)

        err = (QKt_before - QKt_after).abs().max().item()
        assert err < 1e-10, f"float64 logit error {err:.3e} exceeds 1e-10"

    def test_no_bias(self):
        """Invariance holds with zero biases (b_Q = b_K = 0)."""
        torch.manual_seed(7)
        B, T, d = 3, 6, 12
        X = torch.randn(B, T, d)
        W_Q = torch.randn(d, d)
        W_K = torch.randn(d, d)
        b_Q = torch.zeros(d)
        b_K = torch.zeros(d)

        A = _random_invertible(d, scale=0.5, seed=3)
        A_inv = torch.linalg.inv(A)

        QKt_before = _raw_logits(W_Q, W_K, b_Q, b_K, X)
        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        QKt_after = _raw_logits(W_Q2, W_K2, b_Q, b_K, X)

        err = (QKt_before - QKt_after).abs().max().item()
        assert err < 1e-2, f"no-bias logit error {err:.3e}"

    def test_identity_A_is_no_op(self):
        """A = I must leave weights and logits exactly unchanged."""
        torch.manual_seed(0)
        B, T, d = 2, 5, 8
        X = torch.randn(B, T, d)
        W_Q = torch.randn(d, d)
        W_K = torch.randn(d, d)
        b_Q = torch.randn(d)
        b_K = torch.randn(d)

        A = torch.eye(d)
        A_inv = torch.linalg.inv(A)

        QKt_before = _raw_logits(W_Q, W_K, b_Q, b_K, X)
        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        b_Q2 = b_Q @ A
        b_K2 = b_K @ A_inv.T
        QKt_after = _raw_logits(W_Q2, W_K2, b_Q2, b_K2, X)

        err = (QKt_before - QKt_after).abs().max().item()
        assert err == 0.0, f"identity A produced nonzero error {err}"


# ---------------------------------------------------------------------------
# B. Full attention output invariance (nn.MultiheadAttention / TransformerEncoderLayer)
# ---------------------------------------------------------------------------

class TestFullAttentionInvariance:
    """
    Verify that the forward pass of a real PyTorch attention module is unchanged
    after applying AttentionQKGroup.apply_transform.

    nhead=1: any invertible A is valid.
    nhead>1: block-diagonal A (one d_h×d_h block per head) is required.
    """

    def test_single_head_mha_arbitrary_A(self):
        """
        nhead=1: arbitrary A must preserve the full attention output.

        Q and K are not split across heads, so any invertible A works.
        """
        torch.manual_seed(10)
        d_model, nhead = 32, 1
        B, T = 4, 8

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_before, _ = attn(X, X, X, need_weights=True)

        A = _random_invertible(d_model, scale=0.3, seed=5)
        AttentionQKGroup.apply_transform(attn, A)

        with torch.no_grad():
            out_after, _ = attn(X, X, X, need_weights=True)

        err = (out_before - out_after).abs().max().item()
        assert err < 1e-4, (
            f"nhead=1 attention output error {err:.3e} exceeds 1e-4"
        )

    def test_multi_head_mha_block_diagonal_A(self):
        """
        nhead=2: block-diagonal A (one d_h×d_h block per head) must preserve output.
        """
        torch.manual_seed(20)
        d_model, nhead = 32, 2
        d_h = d_model // nhead  # 16
        B, T = 4, 8

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_before, _ = attn(X, X, X, need_weights=True)

        A0 = _random_invertible(d_h, scale=0.2, seed=6)
        A1 = _random_invertible(d_h, scale=0.2, seed=7)
        A = AttentionQKGroup.make_block_diagonal([A0, A1])

        AttentionQKGroup.apply_transform(attn, A)

        with torch.no_grad():
            out_after, _ = attn(X, X, X, need_weights=True)

        err = (out_before - out_after).abs().max().item()
        assert err < 1e-4, (
            f"nhead=2 block-diag attention output error {err:.3e} exceeds 1e-4"
        )

    def test_transformer_encoder_layer_single_head(self):
        """
        TransformerEncoderLayer with nhead=1: apply transform via self_attn,
        verify the layer forward output is unchanged.
        """
        torch.manual_seed(30)
        d_model, nhead, d_ff = 32, 1, 64
        B, T = 3, 6

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        layer.eval()

        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_before = layer(X)

        A = _random_invertible(d_model, scale=0.25, seed=8)
        AttentionQKGroup.apply_transform(layer.self_attn, A)

        with torch.no_grad():
            out_after = layer(X)

        err = (out_before - out_after).abs().max().item()
        assert err < 1e-4, (
            f"TransformerEncoderLayer output error {err:.3e} exceeds 1e-4"
        )

    def test_weights_also_invariant(self):
        """
        Attention weights (softmax output) should also be unchanged with nhead=1.
        """
        torch.manual_seed(40)
        d_model, nhead = 16, 1
        B, T = 2, 5

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            _, weights_before = attn(X, X, X, need_weights=True, average_attn_weights=False)

        A = _random_invertible(d_model, scale=0.2, seed=9)
        AttentionQKGroup.apply_transform(attn, A)

        with torch.no_grad():
            _, weights_after = attn(X, X, X, need_weights=True, average_attn_weights=False)

        err = (weights_before - weights_after).abs().max().item()
        assert err < 1e-4, (
            f"Attention weights error {err:.3e} exceeds 1e-4"
        )

    def test_apply_twice_restores_output(self):
        """
        Applying transform with A then with A_inv restores the original output.
        """
        torch.manual_seed(50)
        d_model, nhead = 32, 1
        B, T = 4, 8

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_original, _ = attn(X, X, X, need_weights=False)

        A = _random_invertible(d_model, scale=0.3, seed=11)
        A_inv = torch.linalg.inv(A)

        AttentionQKGroup.apply_transform(attn, A)
        AttentionQKGroup.apply_transform(attn, A_inv)

        with torch.no_grad():
            out_restored, _ = attn(X, X, X, need_weights=False)

        err = (out_original - out_restored).abs().max().item()
        assert err < 1e-4, (
            f"Round-trip restore error {err:.3e} exceeds 1e-4"
        )

    def test_cross_attention_raises(self):
        """
        Cross-attention with separate projections (in_proj_weight=None) raises ValueError.
        """
        attn = nn.MultiheadAttention(
            embed_dim=16, num_heads=1, kdim=8, vdim=8, batch_first=True
        )
        A = torch.eye(16)
        with pytest.raises(ValueError, match="in_proj_weight"):
            AttentionQKGroup.apply_transform(attn, A)


# ---------------------------------------------------------------------------
# C. Multiple random trials
# ---------------------------------------------------------------------------

class TestMultipleTrials:
    """
    Run raw logit and full output invariance over many seeds and A samples.
    Fail loudly on any instability.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_raw_logit_invariance_many_seeds(self, seed):
        torch.manual_seed(seed)
        B, T, d = 4, 8, 16
        X = torch.randn(B, T, d)
        W_Q = torch.randn(d, d)
        W_K = torch.randn(d, d)
        b_Q = torch.randn(d)
        b_K = torch.randn(d)

        A = _random_invertible(d, scale=0.3, seed=seed + 100)
        A_inv = torch.linalg.inv(A)

        QKt_before = _raw_logits(W_Q, W_K, b_Q, b_K, X)
        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        b_Q2 = b_Q @ A
        b_K2 = b_K @ A_inv.T
        QKt_after = _raw_logits(W_Q2, W_K2, b_Q2, b_K2, X)

        err = (QKt_before - QKt_after).abs().max().item()
        assert err < 1e-2, f"seed={seed}: logit error {err:.3e}"

    @pytest.mark.parametrize("seed", range(5))
    def test_full_output_invariance_many_seeds(self, seed):
        torch.manual_seed(seed)
        d_model, nhead = 32, 1
        B, T = 4, 8

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_before, _ = attn(X, X, X, need_weights=False)

        A = _random_invertible(d_model, scale=0.3, seed=seed + 200)
        AttentionQKGroup.apply_transform(attn, A)

        with torch.no_grad():
            out_after, _ = attn(X, X, X, need_weights=False)

        err = (out_before - out_after).abs().max().item()
        assert err < 1e-4, f"seed={seed}: output error {err:.3e}"

    @pytest.mark.parametrize("seed", range(5))
    def test_multihead_block_diagonal_many_seeds(self, seed):
        torch.manual_seed(seed)
        d_model, nhead = 32, 4
        d_h = d_model // nhead  # 8
        B, T = 3, 6

        attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        X = torch.randn(B, T, d_model)

        with torch.no_grad():
            out_before, _ = attn(X, X, X, need_weights=False)

        blocks = [_random_invertible(d_h, scale=0.15, seed=seed * 10 + h)
                  for h in range(nhead)]
        A = AttentionQKGroup.make_block_diagonal(blocks)
        AttentionQKGroup.apply_transform(attn, A)

        with torch.no_grad():
            out_after, _ = attn(X, X, X, need_weights=False)

        err = (out_before - out_after).abs().max().item()
        assert err < 1e-4, f"seed={seed} nhead=4: output error {err:.3e}"


# ---------------------------------------------------------------------------
# make_block_diagonal unit tests
# ---------------------------------------------------------------------------

class TestMakeBlockDiagonal:
    def test_shape(self):
        d_h, nhead = 8, 4
        blocks = [torch.eye(d_h) for _ in range(nhead)]
        A = AttentionQKGroup.make_block_diagonal(blocks)
        assert A.shape == (d_h * nhead, d_h * nhead)

    def test_identity_blocks_give_identity(self):
        d_h, nhead = 8, 2
        blocks = [torch.eye(d_h) for _ in range(nhead)]
        A = AttentionQKGroup.make_block_diagonal(blocks)
        assert torch.allclose(A, torch.eye(d_h * nhead))

    def test_off_diagonal_zero(self):
        """Blocks must not bleed into off-diagonal quadrants."""
        d_h = 4
        A0 = 2.0 * torch.eye(d_h)
        A1 = 3.0 * torch.eye(d_h)
        A = AttentionQKGroup.make_block_diagonal([A0, A1])
        # Off-diagonal quadrants should be zero
        assert torch.all(A[0:d_h, d_h:] == 0)
        assert torch.all(A[d_h:, 0:d_h] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

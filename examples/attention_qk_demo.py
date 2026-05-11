#!/usr/bin/env python3
"""
Standalone demo: Q/K attention symmetry transform invariance.

Verifies numerically that applying AttentionQKGroup.apply_transform to an
nn.MultiheadAttention module leaves the attention logits and the full
forward-pass output unchanged.

Symmetry
--------
For attention logits  Attn = Q K^T / sqrt(d):

    Q' = Q A,   K' = K A^{-T}   =>   Q' K'^T = Q K^T   (exact)

where A is any invertible d x d matrix.

In weight space (PyTorch: output = x @ W.T + b, W shape (out, in)):

    W_Q' = A^T W_Q       b_Q' = b_Q @ A
    W_K' = A^{-1} W_K    b_K' = b_K @ A^{-T}

Multi-head note: for nhead > 1, A must be block-diagonal (one d_h x d_h block
per head) for the per-head logits Q_h K_h^T to be preserved. For nhead = 1,
any invertible A works.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from symmetry_teleport.groups import AttentionQKGroup


def report(label: str, err: float, tol: float = 1e-4) -> None:
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {label}: max error = {err:.3e}  (tol={tol:.0e})")


def demo_well_conditioned():
    """
    Well-conditioned case: near-identity A = I + 0.01*noise.

    cond(A) is close to 1, so float32 rounding is near the noise floor.
    Demonstrates clean invariance before showing the stress test.
    """
    print("--- Well-conditioned Q/K symmetry (near-identity A) ---")
    torch.manual_seed(42)
    d_model, nhead = 32, 1
    B, T = 4, 8

    # Near-identity A: cond(A) ~ 1.2, errors ~ 1e-4 in float32 with O(1) weights
    A = torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model)
    print(f"  cond(A) = {torch.linalg.cond(A).item():.2f}")

    # Raw Q K^T check with O(1) random weights
    X = torch.randn(B, T, d_model)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(d_model, d_model)
    b_Q = torch.randn(d_model)
    b_K = torch.randn(d_model)
    A_inv = torch.linalg.inv(A)

    Q = X @ W_Q.T + b_Q
    K = X @ W_K.T + b_K
    QKt_before = Q @ K.transpose(-2, -1)
    QKt_after = (X @ (A.T @ W_Q).T + b_Q @ A) @ (
        (X @ (A_inv @ W_K).T + b_K @ A_inv.T).transpose(-2, -1)
    )
    report("Q K^T invariance [float32]",
           (QKt_before - QKt_after).abs().max().item(), tol=5e-4)

    # Full MHA forward check (xavier init → even smaller errors)
    attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
    with torch.no_grad():
        out_before, _ = attn(X, X, X, need_weights=False)
    AttentionQKGroup.apply_transform(attn, A)
    with torch.no_grad():
        out_after, _ = attn(X, X, X, need_weights=False)
    report("attention output    [float32]",
           (out_before - out_after).abs().max().item(), tol=1e-4)


def demo_raw_logit_stress():
    """
    Float32 stress test: ill-conditioned A = I + 0.3*noise.

    With cond(A) ~ 100-200, float32 rounding accumulates through A^{-1}.
    The residual ~1e-3 is expected from ||Q||·cond(A)·ε_mach·d·||K||,
    not from the transform being wrong — float64 confirms near-machine-precision.
    """
    print("\n--- Float32 stress test (ill-conditioned A, seed=0) ---")
    torch.manual_seed(0)
    d = 32
    B, T = 4, 8

    for dtype, tol, tag in [
        (torch.float32, 1e-2, "float32"),
        (torch.float64, 1e-10, "float64"),
    ]:
        X = torch.randn(B, T, d, dtype=dtype)
        W_Q = torch.randn(d, d, dtype=dtype)
        W_K = torch.randn(d, d, dtype=dtype)
        b_Q = torch.randn(d, dtype=dtype)
        b_K = torch.randn(d, dtype=dtype)

        A = (torch.eye(d) + 0.3 * torch.randn(d, d)).to(dtype)
        A_inv = torch.linalg.inv(A)
        print(f"  cond(A) [{tag}] = {torch.linalg.cond(A).item():.2f}")

        Q = X @ W_Q.T + b_Q
        K = X @ W_K.T + b_K
        QKt_before = Q @ K.transpose(-2, -1)

        W_Q2 = A.T @ W_Q
        W_K2 = A_inv @ W_K
        b_Q2 = b_Q @ A
        b_K2 = b_K @ A_inv.T
        QKt_after = (X @ W_Q2.T + b_Q2) @ (X @ W_K2.T + b_K2).transpose(-2, -1)

        err = (QKt_before - QKt_after).abs().max().item()
        report(f"Q K^T invariance [{tag}]", err, tol=tol)


def demo_single_head_full_output():
    """nhead=1: full attention output must be unchanged for arbitrary A."""
    print("\n--- Full attention output, nhead=1, arbitrary A ---")
    torch.manual_seed(1)
    d_model, nhead = 32, 1
    B, T = 4, 8

    attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
    X = torch.randn(B, T, d_model)

    A = torch.eye(d_model) + 0.3 * torch.randn(d_model, d_model)
    print(f"  cond(A) = {torch.linalg.cond(A).item():.2f}")

    with torch.no_grad():
        out_before, w_before = attn(X, X, X, need_weights=True)

    AttentionQKGroup.apply_transform(attn, A)

    with torch.no_grad():
        out_after, w_after = attn(X, X, X, need_weights=True)

    report("attention output", (out_before - out_after).abs().max().item())
    report("attention weights", (w_before - w_after).abs().max().item())


def demo_multi_head_block_diagonal():
    """nhead=4: block-diagonal A (one d_h x d_h block per head)."""
    print("\n--- Full attention output, nhead=4, block-diagonal A ---")
    torch.manual_seed(2)
    d_model, nhead = 32, 4
    d_h = d_model // nhead  # 8
    B, T = 4, 8

    attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
    X = torch.randn(B, T, d_model)

    blocks = [torch.eye(d_h) + 0.2 * torch.randn(d_h, d_h) for _ in range(nhead)]
    A = AttentionQKGroup.make_block_diagonal(blocks)
    print(f"  A shape: {list(A.shape)}, cond(A) = {torch.linalg.cond(A).item():.2f}")

    with torch.no_grad():
        out_before, _ = attn(X, X, X, need_weights=False)

    AttentionQKGroup.apply_transform(attn, A)

    with torch.no_grad():
        out_after, _ = attn(X, X, X, need_weights=False)

    report("attention output", (out_before - out_after).abs().max().item())


def demo_transformer_encoder_layer():
    """TransformerEncoderLayer: transform self_attn, full layer output unchanged."""
    print("\n--- TransformerEncoderLayer forward, nhead=1 ---")
    torch.manual_seed(3)
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

    A = torch.eye(d_model) + 0.25 * torch.randn(d_model, d_model)

    with torch.no_grad():
        out_before = layer(X)

    AttentionQKGroup.apply_transform(layer.self_attn, A)

    with torch.no_grad():
        out_after = layer(X)

    report("layer output", (out_before - out_after).abs().max().item())


def demo_inverse_restores():
    """Apply A then A^{-1}: output must be restored."""
    print("\n--- Round-trip A then A^{-1} ---")
    torch.manual_seed(4)
    d_model, nhead = 32, 1
    B, T = 4, 8

    attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
    X = torch.randn(B, T, d_model)

    A = torch.eye(d_model) + 0.3 * torch.randn(d_model, d_model)
    A_inv = torch.linalg.inv(A)

    with torch.no_grad():
        out_original, _ = attn(X, X, X, need_weights=False)

    AttentionQKGroup.apply_transform(attn, A)
    AttentionQKGroup.apply_transform(attn, A_inv)

    with torch.no_grad():
        out_restored, _ = attn(X, X, X, need_weights=False)

    report("round-trip output", (out_original - out_restored).abs().max().item())


if __name__ == "__main__":
    print("AttentionQKGroup invariance demo")
    print("=" * 50)
    demo_well_conditioned()
    demo_raw_logit_stress()
    demo_single_head_full_output()
    demo_multi_head_block_diagonal()
    demo_transformer_encoder_layer()
    demo_inverse_restores()
    print("\nDone.")

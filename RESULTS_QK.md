# FFN+QK Teleportation Benchmark Results

## Goal

Determine whether adding diagonal Q/K attention scaling to the FFN teleportation
step (FFN+QK) improves convergence compared to FFN-only teleportation and
baseline SGD on a character-level language modeling task.

---

## Symmetry transforms

### FFN diagonal scaling (existing)

For `y = W2 · ReLU(W1 x + b1) + b2`, positive diagonal scales `s ∈ ℝ^{d_ff}`:

```
W1' = diag(s) W1    b1' = s ⊙ b1
W2' = W2 diag(1/s)  b2' = b2
```

Preserves output exactly for ReLU. Does NOT preserve output for GELU.

### Q/K diagonal scaling (new)

For `MultiheadAttention` with packed `in_proj_weight` and positive diagonal
scales `a ∈ ℝ^d` (flat; block-diagonal per head):

```
W_Q rows i  ×= a[i]    b_Q ×= a
W_K rows i  /= a[i]    b_K /= a
W_V, b_V    unchanged
```

Preserves `Q K^T = (diag(a) W_Q x)(diag(1/a) W_K x)^T` exactly. Since the softmax weights are unchanged and the V path is unchanged, the attention output is preserved under this transform.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Task | Character-level next-token prediction (Alice corpus) |
| Model | `CharLMTransformer` (d=32, nhead=2, d_ff=64, 1 layer, ReLU) |
| Loss | Cross-entropy (flat token prediction) |
| Optimizer | SGD, lr=0.05 |
| Steps | 100 |
| Batch size | 16 |
| Teleport every | 5 steps |
| Inner steps | 5 |
| Restarts | 10 (FFN), 10 (QK) |
| log_s/a clip | (−2, 2) |
| Seeds | 0, 1, 2 (paired init) |

All three conditions (Baseline / FFN-only / FFN+QK) start from the same
`state_dict` snapshot per seed. Minibatch indices are identical across arms.

---

## Results

<!-- python scripts/bench_ffn_qk_compare.py --seeds 0 1 2 --steps 100 --inner-steps 5 -->

### Per-seed final loss

| Seed | Baseline | FFN-only | FFN+QK |
|------|----------|----------|--------|
| 0 | 2.7587 | 2.7421 | 2.7401 |
| 1 | 2.6539 | 2.6534 | 2.6516 |
| 2 | 2.7376 | 2.7317 | 2.7307 |
| **Mean** | **2.7167** | **2.7091** | **2.7075** |

### Area Under Loss Curve (AUC, lower is better)

| Seed | Baseline | FFN-only | FFN+QK | ΔAUC (ffn vs base) | ΔAUC (ffn+qk vs base) |
|------|----------|----------|--------|--------------------|-----------------------|
| 0 | 303.34 | 302.63 | 302.50 | −0.71 | −0.84 |
| 1 | 309.69 | 308.58 | 308.42 | −1.11 | −1.27 |
| 2 | 307.50 | 306.49 | 306.43 | −1.01 | −1.07 |
| **Mean** | **306.84** | **305.90** | **305.78** | **−0.94** | **−1.06** |

FFN+QK vs FFN-only mean ΔAUC: **−0.12**

### Teleportation acceptance

| Condition | Attempts/seed | Accepted | Rate |
|-----------|--------------|----------|------|
| FFN-only | 20 | 20/20 | 100% |
| FFN+QK | 20 | 20/20 | 100% |

### Loss trajectory (mean across seeds)

| Step | Baseline | FFN-only | FFN+QK |
|------|----------|----------|--------|
| 0 | 3.7893 | 3.7893 | 3.7893 |
| 10 | 3.5424 | 3.5420 | 3.5420 |
| 20 | 3.2852 | 3.2824 | 3.2823 |
| 30 | 3.2594 | 3.2541 | 3.2539 |
| 40 | 3.0591 | 3.0503 | 3.0496 |
| 50 | 2.9296 | 2.9170 | 2.9163 |
| 60 | 3.1858 | 3.1739 | 3.1725 |
| 70 | 2.8090 | 2.7945 | 2.7932 |
| 80 | 2.9151 | 2.9071 | 2.9043 |
| 90 | 2.8660 | 2.8491 | 2.8446 |

---

## Interpretation

Both teleportation conditions consistently outperform baseline SGD in AUC across
all three seeds. FFN+QK provides a small additional improvement over FFN-only.

**FFN-only vs baseline:** −0.94 AUC (mean). Consistent across seeds (−0.71,
−1.11, −1.01). Teleportation finds scalings that improve the virtual-SGD
landscape and is accepted 100% of the time on this task.

**FFN+QK vs FFN-only:** −0.12 AUC (mean). Small but consistent across seeds
(−0.13, −0.16, −0.06). The Q/K diagonal scaling is a true output-preserving
symmetry — it changes gradient geometry without moving on the loss surface —
so any benefit comes purely from a better-conditioned parameter space for
subsequent SGD steps.

**Acceptance:** 100% in both conditions. The task and architecture combination
appears favorable for the virtual-SGD improvement criterion. Low acceptance
rates would be expected in settings where the loss landscape is already
well-conditioned or the search space (diagonal only) is too restricted.

---

## Limitations

- **Small corpus and short run:** 100 SGD steps on ~750 characters is not a
  full convergence regime. Results may not generalize to larger settings.
- **CPU only:** Wall-clock overhead of FFN+QK (roughly 2× FFN-only per teleport
  attempt) is not reported here; all comparisons are in SGD-step space.
- **Diagonal restriction:** Only diagonal a is searched; the full GL(d) Q/K
  symmetry group is not explored.
- **Single layer:** Teleportation is applied to layer 0 only.
- **Q/K is a true symmetry:** Because Q K^T is invariant, the instantaneous
  loss is unchanged by the transform. Improvement can only come from gradient
  geometry changes that help subsequent SGD steps. This is subtler than the
  FFN case and may require more steps or larger models to see consistent benefit.

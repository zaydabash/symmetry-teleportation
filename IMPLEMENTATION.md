# Implementation Notes

Factual description of the package design, training integration, symmetry
theory, objective/acceptance logic, and the included language modeling task.

---

## 1. Package structure

```
symmetry_teleport/
    __init__.py      exports TeleportSGD, ScalarRescalingGroup, AttentionQKGroup
    optim.py         TeleportSGD: SGD + periodic teleport scheduling
    teleport.py      teleport_ffn_diagonal, teleport_qk_diagonal, helpers
    groups.py        ScalarRescalingGroup, AttentionQKGroup: in-place transforms
    utils.py         shared helpers (seeds, tensor hashing)
examples/
    tiny_transformer_example.py    synthetic regression demo
    text_language_task_example.py  char-level language modeling demo
    attention_qk_demo.py           Q/K symmetry invariance demo
scripts/
    bench_tiny_transformer.py      paired multi-seed FFN benchmark
    bench_ffn_qk_compare.py        3-condition FFN vs FFN+QK benchmark
tests/
    test_optimizer_basic.py
    test_group_scalar.py
    test_new_features.py
    test_attention_qk.py
    test_teleport_qk.py
```

The package has no runtime dependencies beyond `torch` and `numpy`.

---

## 2. Training integration

`TeleportSGD` is a drop-in replacement for `torch.optim.SGD`. The model must
implement `get_ffn_layers(layer_idx)` for FFN-only teleportation.  For
`teleport_target='ffn_qk'` it must also implement `get_attn_layer(layer_idx)`:

```python
def get_ffn_layers(self, layer_idx: int) -> tuple[nn.Linear, nn.Linear]:
    ...

def get_attn_layer(self, layer_idx: int) -> nn.MultiheadAttention:
    ...
```

Minimal wiring:

```python
optimizer = TeleportSGD(
    model.parameters(),
    lr=0.01,
    teleport_every=5,
    teleport_config={
        "model": model,
        "layer_idx": 0,
        "X_teleport": X_ref,       # reference batch for inner search
        "Y_teleport": Y_ref,
        "loss_fn": loss_fn,
        "lr_theta": 0.3,
        "inner_steps": 50,
        "objective": "virtual_sgd_improve",
        "s_param": "projected",
        "log_s_clip": (-2.0, 2.0),
    },
)
```

`optimizer.step()` performs a standard SGD update, then calls
`_apply_teleportation()` every `teleport_every` steps. The teleportation
search uses `torch.func.functional_call` and never mutates model parameters
during the search. Parameters are restored if the acceptance criterion fails.

---

## 3. Symmetry type

The implementation exploits **diagonal scaling symmetry** of two-layer FFNs
with ReLU activation.

For the FFN block `y = W2 · activation(W1 · x + b1) + b2`, the transformation
parameterized by `s ∈ ℝ_{>0}^{d_ff}` is:

```
W1' = diag(s) @ W1      (row-wise scale)
b1' = s ⊙ b1            (element-wise scale)
W2' = W2 @ diag(1/s)    (column-wise inverse scale)
b2' = b2                 (unchanged)
```

For ReLU: `activation(s · z) = s · activation(z)` when `s > 0`. This means
`W2' · activation(W1' · x + b1') = W2 · activation(W1 · x + b1)`, so the
network output — and therefore the loss — is unchanged. The transformed point
`(W1', b1', W2')` lies on the same loss contour as `(W1, b1, W2)`.

`ScalarRescalingGroup.apply_transform(linear1, linear2, s)` applies this
transform in-place using `mul_`.

---

## 4. Objective and acceptance logic

**Inner search objective (`virtual_sgd_improve`)**

For each teleportation attempt, an inner optimizer searches over `s` to
minimize the loss after `k = 5` virtual SGD steps from the transformed
parameters, at learning rate `lr_virtual = 2 × lr`:

```
J(s) = L(θ(s) − lr_virtual · ∇L(θ(s)) − lr_virtual · ∇L(…))
```

where `θ(s)` denotes the parameters after applying the diagonal transform
with scaling vector `s`. The search runs 10 random restarts (noise scale
`5e-2` around identity) to avoid identity lock.

**Parameterization**

With `s_param="projected"`, `s` is optimized directly in
`[exp(log_s_clip[0]), exp(log_s_clip[1])]` via projected gradient descent.

**Acceptance criterion**

After the inner search returns the best `s*`, a separate stateful virtual SGD
rollout re-evaluates both paths using the actual optimizer state:

```
accepted  iff  L_tp_virtual < L_baseline_virtual − ε
```

where `ε = 1e-10`. If rejected, parameters are restored exactly via
`model.load_state_dict`. A teleportation is additionally classified as
`nontrivial` if `max|log s*| ≥ 1e-6` and `changed` if `max|Δparam| ≥ 1e-6`.

---

## 5. Benchmark pairing

The multi-seed benchmark (`scripts/bench_tiny_transformer.py`) uses a
strictly paired design:

- Both the baseline and teleport model are initialized from the same
  `state_dict` snapshot per seed.
- Minibatches are drawn with `torch.arange(step * B, (step+1) * B) % N`,
  giving identical sequence for both arms.
- A `data_hash` (MD5 of `X.numpy().tobytes()`) and `param_hash` (MD5 of
  all parameter bytes at init) are logged to confirm pairing.
- Loss AUC is computed as `numpy.trapz(losses, dx=1.0)` over all steps.

---

## 6. Real text language task

**Script:** `examples/text_language_task_example.py`

**Task:** Character-level next-token prediction. Given a sequence of `T = 32`
characters, predict the next character at every position (autoregressive
teacher-forced training).

**Corpora (--corpus flag):**

| Flag | Source | Length | Vocab |
|------|--------|--------|-------|
| `alice` (default) | *Alice's Adventures in Wonderland* (Lewis Carroll, 1865, public domain) | ~754 chars | 37 |
| `shakespeare` | *Hamlet* Act III Scene I soliloquy (William Shakespeare, c. 1600, public domain) | ~600 chars | 32 |

Both texts are bundled as string literals; no external files or downloads are
required.

**Vocabulary:** Unique characters in the selected text. Encoded as integer
indices; no tokenizer library needed.

**Model:** `CharLMTransformer` — `nn.Embedding` + learned position embedding
→ `nn.TransformerEncoder` (1 layer, ReLU FFN, causal attention mask) →
`nn.Linear` LM head. Implements `get_ffn_layers(0)` returning the FFN's
`(linear1, linear2)`.

Each forward pass adds a learned position embedding (one vector per sequence
position) to the character embedding before the encoder. A causal upper-
triangular additive mask (`-inf` above the diagonal) is passed to
`TransformerEncoder.forward`, ensuring that position `t` attends only to
positions `0 … t`. There is no future-token leakage.

**Loss:** `F.cross_entropy(logits.view(-1, V), targets.view(-1))` wrapped in
`LMCrossEntropyLoss` so `loss_fn(model(X), Y)` works directly in the
teleport config.

**Smoke test:** Pass `--smoke` to run in ~10 seconds on CPU (30 steps,
`inner_steps=10`). Omit for a 200-step paired comparison.

**Why character-level is the smallest valid real-text task:**

- No tokenizer dependency (one `dict` lookup per character).
- No external data files (text is a string literal).
- CrossEntropyLoss is a direct classification loss — same mathematical
  structure as any other training objective.
- The FFN uses ReLU, so the diagonal scaling symmetry is exact.

---

## 7. Q/K attention diagonal scaling symmetry

For `teleport_target='ffn_qk'`, an additional diagonal per-head scaling is
applied to the Q and K projections alongside the FFN scaling.

**Why diagonal:** The full Q/K symmetry group is GL(d), i.e. any invertible
matrix A such that Q' K'^T = (QA)(KA^{-T})^T = QK^T. A dense A requires
matrix inversion (unstable for large cond(A)) and O(d²) parameters. The
diagonal restriction A = diag(a) keeps the search in O(d) parameters, avoids
inversion (a_inv = 1/a element-wise), and mirrors the FFN diagonal symmetry.
For multi-head attention the diagonal must be block-diagonal (one d_h × d_h
block per head); a flat per-element diagonal is the natural restriction.

**Weight-space transform** (no matrix inverse; operations are element-wise):

```
W_Q rows i  ×= a[i]      W_K rows i  /= a[i]
b_Q         ×= a          b_K         /= a
W_V, b_V    unchanged
```

**Why the full output is unchanged:** Softmax(QK^T / √d) is invariant to
diagonal scaling since Q'K'^T = QK^T. The value output V is not scaled.
Therefore the full attention output — and the entire model output — is
preserved. This is a **stronger symmetry** than the FFN case, where the
output is preserved at the current input but the gradient geometry changes.

**Parameterization:** `a = exp(log_a)`, `log_a ∈ ℝ^d` initialized near zero.
The inner search uses the same `virtual_sgd_improve` objective as the FFN.
Flash attention on CPU does not support `create_graph=True` when
`in_proj_weight` has `requires_grad=True`; `teleport_qk_diagonal` therefore
wraps its optimization loop with `sdpa_kernel(SDPBackend.MATH)`.

**Acceptance:** The combined FFN+QK transform is evaluated jointly via
`_compute_virtual_sgd_losses_ffn_qk`. Both transforms are accepted or rejected
together; a rejection restores exact original parameters.

**Limitation:** Requires `nn.MultiheadAttention` with packed `in_proj_weight`
(`kdim == vdim == embed_dim`). Cross-attention (separate `Q`, `K`, `V`
weight matrices) is not currently supported.

---

## 8. Limitations

- **Single layer:** Only one layer is teleported per step (`layer_idx` is
  fixed). Multi-layer teleportation is not implemented.
- **No wall-clock guarantees:** Teleportation adds inner optimization overhead
  proportional to `inner_steps × virtual_steps × batch_size`. Reported
  results are in SGD-step space (AUC, steps-to-threshold), not wall-clock
  time. FFN+QK is roughly 2× slower per teleport attempt than FFN-only.
- **Activation requirement:** The FFN diagonal scaling symmetry is exactly
  loss-invariant only for ReLU. GELU is not positively homogeneous.
- **Reference batch:** `X_teleport` / `Y_teleport` are fixed at optimizer
  construction. Using a stale reference batch may reduce teleportation quality.
- **Q/K scope:** The diagonal restriction on A is conservative. Dense GL(d)
  transforms would explore a larger symmetry group but require stable matrix
  inversion.

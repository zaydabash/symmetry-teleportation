# Symmetry Teleportation

A PyTorch optimizer that applies loss invariant symmetry transformations during
training to reduce the number of SGD steps needed for optimization in supported
models.

Based on: Bo Zhao, Nima Dehmamy, Robin Walters, Rose Yu.
*Symmetry Teleportation for Accelerated Optimization*. NeurIPS 2022.

> **Scope:** Two symmetry types are implemented:
>
> - **FFN diagonal scaling** (default): loss-invariant for ReLU FFNs.
>   Model must implement `get_ffn_layers(layer_idx) → (linear1, linear2)`.
> - **Q/K attention scaling** (`teleport_target='ffn_qk'`): diagonal
>   per-head scaling that preserves Q K^T logits exactly.
>   Model must additionally implement `get_attn_layer(layer_idx) → nn.MultiheadAttention`.

## Minimal usage

```python
from symmetry_teleport import TeleportSGD

optimizer = TeleportSGD(
    model.parameters(),
    lr=0.01,
    teleport_every=5,
    teleport_config={
        "model": model,
        "layer_idx": 0,
        "X_teleport": X,
        "Y_teleport": Y,
        "loss_fn": loss_fn,
        "lr_theta": 0.3,
        "inner_steps": 50,
        "objective": "virtual_sgd_improve",
        "s_param": "projected",
    },
)

for X_batch, Y_batch in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(model(X_batch), Y_batch)
    loss.backward()
    optimizer.step()

stats = optimizer.teleport_stats()
print(f"Accepted {stats['accepted_count']} / {stats['total_attempts']} attempts")
```

## Examples

Synthetic regression (quick demo):

```bash
python3 examples/tiny_transformer_example.py
```

Character-level language modeling on a bundled public-domain text snippet
(no external data required):

```bash
python3 examples/text_language_task_example.py          # full run
python3 examples/text_language_task_example.py --smoke  # smoke test (~10s on CPU)
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for details on the package design,
symmetry type, objective/acceptance logic, and the text task.

See [RESULTS.md](RESULTS.md) for paired multi-seed benchmark results on both
text corpora.

See [RESULTS_QK.md](RESULTS_QK.md) for the 3-condition (Baseline / FFN-only
/ FFN+QK) benchmark on the Alice corpus.

## Benchmark scripts

```bash
python3 scripts/bench_tiny_transformer.py --help
python3 scripts/bench_ffn_qk_compare.py --smoke       # 3-condition smoke test
python3 scripts/bench_ffn_qk_compare.py --seeds 0 1 2 # full 3-seed run
```

## Q/K attention teleportation

To also teleport the Q/K projections alongside the FFN, set
`teleport_target='ffn_qk'` (requires `objective='virtual_sgd_improve'`):

```python
optimizer = TeleportSGD(
    model.parameters(),
    lr=0.01,
    teleport_every=5,
    teleport_config={
        "model": model,
        "layer_idx": 0,
        "X_teleport": X,
        "Y_teleport": Y,
        "loss_fn": loss_fn,
        "lr_theta": 0.3,
        "inner_steps": 30,
        "objective": "virtual_sgd_improve",
        "s_param": "projected",
        "teleport_target": "ffn_qk",  # add Q/K diagonal scaling
    },
)
```

The Q/K transform `(W_Q, b_Q, W_K, b_K) → (diag(a) W_Q, a ⊙ b_Q, diag(1/a) W_K, b_K / a)`
preserves Q K^T exactly for any `a > 0`. The acceptance decision evaluates the
combined FFN+QK transform via the same virtual-SGD rollout used for FFN-only.

## Notes and limitations

- Teleportation adds inner optimization overhead per attempt.
- Results in this repo are reported as fewer SGD steps or lower loss AUC, not
  wall clock speedup.
- The benchmarked configuration uses `objective="virtual_sgd_improve"` and
  `s_param="projected"`.
- Q/K teleportation requires `nn.MultiheadAttention` with packed `in_proj_weight`
  (the default; `kdim == vdim == embed_dim`).

## Installation

```bash
pip install -e /path/to/symmetryteleportationoptimization
```

## Tests

```bash
python3 -m pytest -q
```

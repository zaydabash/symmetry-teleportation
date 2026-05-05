# Symmetry Teleportation

A PyTorch optimizer that applies loss invariant symmetry transformations during
training to reduce the number of SGD steps needed for optimization in supported
models.

Based on: Bo Zhao, Nima Dehmamy, Robin Walters, Rose Yu.
*Symmetry Teleportation for Accelerated Optimization*. NeurIPS 2022.

> **Scope:** The current implementation focuses on diagonal scaling symmetry
> for transformer FFN layers with ReLU activation. The symmetry is
> loss-invariant for ReLU; it does not hold exactly for GELU. The model must
> implement `get_ffn_layers(layer_idx)` and return `(linear1, linear2)`.

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

## Benchmark script

```bash
python3 scripts/bench_tiny_transformer.py --help
```

The benchmark script compares baseline SGD and teleportation on a paired tiny
transformer setup and writes results to `results/`.

## Notes and limitations

- Teleportation adds inner optimization overhead per attempt.
- Results in this repo are reported as fewer SGD steps or lower loss AUC, not
  wall clock speedup.
- The current implementation supports diagonal FFN scaling symmetry only.
- The benchmarked configuration uses `objective="virtual_sgd_improve"` and
  `s_param="projected"`.

## Installation

```bash
pip install -e /path/to/symmetryteleportationoptimization
```

## Tests

```bash
python3 -m pytest -q
```

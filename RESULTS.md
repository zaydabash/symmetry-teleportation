# Experiment Results

Small-scale paired evaluation of TeleportSGD vs baseline SGD on real
character-level language modeling tasks. Results are in SGD-step space;
no wall-clock comparisons are made.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | `CharLMTransformer` (d_model=32, nhead=2, dim_feedforward=64, 1 layer) |
| Activation | ReLU (required for exact diagonal scaling symmetry) |
| Task | Causal char-level next-token prediction (seq_len=32) |
| Loss | Cross-entropy averaged over all positions |
| Optimizer | SGD (baseline) vs TeleportSGD (same lr, same minibatches) |
| lr | 0.05 |
| batch_size | 16 |
| steps | 150 |
| teleport_every | 5 |
| inner_steps | 10 (inner Adam iterations per teleport attempt) |
| virtual_steps | 5 (virtual SGD steps in acceptance criterion) |
| restarts | 10 (random restarts per teleport search) |
| Pairing | Both arms share identical parameter init and minibatch sequence per seed |
| Metric | Loss AUC = `numpy.trapz(losses)` over all 150 steps |
| ΔAUC | teleport_auc − baseline_auc (negative = teleport is better) |

**Why 150 steps:** 300 steps with `inner_steps=20` required ~19 min/run
(measured: 1127 s). At that rate 8 runs would take ~2.5 hours. 150 steps at
`inner_steps=10` (~13 min/run) is the largest feasible run size in this
session.

---

## Results

### Alice corpus (*Alice's Adventures in Wonderland*, Lewis Carroll 1865)

Text: 754 chars, vocab: 37 unique chars, sequences: 722

| Seed | Baseline AUC | Teleport AUC | ΔAUC | Base final loss | TP final loss | Accepted / Attempts |
|------|-------------|-------------|------|----------------|--------------|---------------------|
| 0 | 438.47 | 435.15 | −3.32 | 2.7558 | 2.6874 | 28 / 30 |
| 1 | 447.73 | 444.05 | −3.67 | 2.7331 | 2.7699 | 29 / 30 |
| 2 | 445.47 | 442.69 | −2.78 | 2.670 | 2.667 | 29 / 30 |
| 3 | 446.47 | 442.94 | −3.53 | 2.771 | 2.759 | 29 / 30 |
| 4 | 447.52 | 444.73 | −2.79 | 2.806 | 2.809 | 28 / 30 |

Note: Seeds 1 and 4 have a slightly higher teleport final loss despite lower
teleport AUC. This is an honest result — the AUC improvement shows lower
average loss across all 150 steps, but the final point can fluctuate.

**Aggregate (Alice, n=5):**

| Metric | Value |
|--------|-------|
| Mean ΔAUC | −3.22 |
| Std ΔAUC | 0.41 |
| Wins (ΔAUC < 0) | 5 / 5 |
| Mean baseline final loss | 2.747 |
| Mean teleport final loss | 2.738 |
| Total accepted / attempts | 143 / 150 (95.3 %) |

---

### Shakespeare corpus (*Hamlet* Act III Scene I, William Shakespeare c. 1600)

Text: 600 chars, vocab: 32 unique chars, sequences: 568

| Seed | Baseline AUC | Teleport AUC | ΔAUC | Base final loss | TP final loss | Accepted / Attempts |
|------|-------------|-------------|------|----------------|--------------|---------------------|
| 0 | 435.58 | 431.57 | −4.01 | 2.634 | 2.535 | 30 / 30 |
| 1 | 432.15 | 429.71 | −2.44 | 2.640 | 2.609 | 30 / 30 |
| 2 | 428.81 | 426.60 | −2.21 | 2.666 | 2.620 | 30 / 30 |

**Aggregate (Shakespeare, n=3):**

| Metric | Value |
|--------|-------|
| Mean ΔAUC | −2.89 |
| Std ΔAUC | 0.98 |
| Wins (ΔAUC < 0) | 3 / 3 |
| Mean baseline final loss | 2.647 |
| Mean teleport final loss | 2.588 |
| Total accepted / attempts | 90 / 90 (100.0 %) |

---

## Summary Table

| Corpus | Seeds | Steps | Mean base final loss | Mean TP final loss | Mean ΔAUC | Std ΔAUC | Wins | Accepted / Total |
|--------|-------|-------|---------------------|-------------------|-----------|---------|------|-----------------|
| Alice (LM) | 5 | 150 | 2.747 | 2.738 | −3.22 | 0.41 | 5/5 | 143/150 |
| Shakespeare (LM) | 3 | 150 | 2.647 | 2.588 | −2.89 | 0.98 | 3/3 | 90/90 |

---

## Notes

- **Negative ΔAUC means teleport is better**: ΔAUC = teleport_AUC −
  baseline_AUC; smaller AUC = lower cumulative loss over training.
- **SGD-step comparisons only**: teleportation adds per-attempt inner
  optimization overhead. No wall-clock speedup is claimed.
- **Small scale**: two bundled text snippets (~600–750 chars each), one
  transformer layer, CPU only. Results are preliminary evidence, not a
  production benchmark.
- **No hyperparameter tuning**: lr, batch_size, teleport_every, d_model,
  and all model hyperparameters are unchanged from the original script
  defaults. `inner_steps` was reduced from the script default (20) to 10
  solely to keep run time feasible (~13 min/run vs ~19 min/run).
- **Acceptance rate**: 95.3% (Alice) and 100% (Shakespeare) indicate the
  inner search consistently found improvements under the virtual-SGD
  acceptance criterion.
- **No statistical test**: "wins" counts are descriptive. With 5 and 3
  seeds respectively, no formal significance test is reported.

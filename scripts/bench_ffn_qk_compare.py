#!/usr/bin/env python3
"""
3-condition benchmark: Baseline SGD vs FFN-only teleport vs FFN+QK teleport.

Runs on the Alice corpus using a character-level language model.

Usage
-----
Smoke test (fast CI check, ~30s):
    python scripts/bench_ffn_qk_compare.py --smoke

Full run (3 seeds, 100 steps):
    python scripts/bench_ffn_qk_compare.py --seeds 0 1 2 --steps 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from symmetry_teleport import TeleportSGD  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus (same as text_language_task_example)
# ---------------------------------------------------------------------------

_CORPUS = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it, and what "
    "is the use of a book, thought Alice, without pictures or conversations? So "
    "she was considering in her own mind, as well as she could, for the hot day "
    "made her feel very sleepy and stupid, whether the pleasure of making a "
    "daisy-chain would be worth the trouble of getting up and picking the daisies,"
    " when suddenly a White Rabbit with pink eyes ran close by her. There was "
    "nothing so very remarkable in that; nor did Alice think it so very much out "
    "of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be "
    "late!"
)


# ---------------------------------------------------------------------------
# Minimal CharLM model
# ---------------------------------------------------------------------------

def _build_vocab(text):
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}


def _make_sequences(text, stoi, seq_len):
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = len(tokens) - seq_len
    X = torch.stack([tokens[i:i + seq_len] for i in range(n)])
    Y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in range(n)])
    return X, Y


class _LMLoss(nn.Module):
    def forward(self, logits, targets):
        V = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


class _CharLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, d_ff=64, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.0, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        return self.head(self.encoder(self.embed(x) + self.pos(pos), mask=mask))

    def get_ffn_layers(self, layer_idx=0):
        enc = self.encoder.layers[layer_idx]
        return enc.linear1, enc.linear2

    def get_attn_layer(self, layer_idx=0):
        return self.encoder.layers[layer_idx].self_attn


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _batch_idx(step, batch_size, n):
    return torch.arange(step * batch_size, (step + 1) * batch_size) % n


def _run_baseline(model, X, Y, loss_fn, lr, steps, batch_size):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = _batch_idx(step, batch_size, len(X))
        opt.zero_grad()
        loss = loss_fn(model(X[idx]), Y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _teleport_config(model, X, Y, loss_fn, inner_steps, target='ffn'):
    cfg = {
        'model': model,
        'layer_idx': 0,
        'X_teleport': X,
        'Y_teleport': Y,
        'loss_fn': loss_fn,
        'lr_theta': 0.3,
        'inner_steps': inner_steps,
        'objective': 'virtual_sgd_improve',
        's_param': 'projected',
        'log_s_clip': (-2.0, 2.0),
        'teleport_target': target,
    }
    return cfg


def _run_teleport(model, X, Y, loss_fn, lr, steps, batch_size,
                  teleport_every, inner_steps, target):
    cfg = _teleport_config(model, X, Y, loss_fn, inner_steps, target)
    opt = TeleportSGD(
        model.parameters(), lr=lr,
        teleport_every=teleport_every,
        teleport_config=cfg,
    )
    losses = []
    for step in range(steps):
        idx = _batch_idx(step, batch_size, len(X))
        opt.zero_grad()
        loss = loss_fn(model(X[idx]), Y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses, opt.teleport_stats()


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(seed, steps, inner_steps, teleport_every, batch_size, lr):
    stoi = _build_vocab(_CORPUS)
    vocab_size = len(stoi)
    seq_len = 32
    X, Y = _make_sequences(_CORPUS, stoi, seq_len)
    loss_fn = _LMLoss()

    torch.manual_seed(seed)
    m_base = _CharLM(vocab_size, d_model=32, nhead=2, d_ff=64, max_len=seq_len)
    state0 = {k: v.clone() for k, v in m_base.state_dict().items()}

    m_ffn = _CharLM(vocab_size, d_model=32, nhead=2, d_ff=64, max_len=seq_len)
    m_ffn.load_state_dict(state0)
    m_fq = _CharLM(vocab_size, d_model=32, nhead=2, d_ff=64, max_len=seq_len)
    m_fq.load_state_dict(state0)

    print(f"  seed={seed}: baseline...", flush=True)
    base = _run_baseline(m_base, X, Y, loss_fn, lr, steps, batch_size)

    print(f"  seed={seed}: ffn-only...", flush=True)
    ffn, ffn_stats = _run_teleport(
        m_ffn, X, Y, loss_fn, lr, steps, batch_size,
        teleport_every, inner_steps, target='ffn',
    )

    print(f"  seed={seed}: ffn+qk...", flush=True)
    fq, fq_stats = _run_teleport(
        m_fq, X, Y, loss_fn, lr, steps, batch_size,
        teleport_every, inner_steps, target='ffn_qk',
    )

    return {
        'seed': seed,
        'base': base, 'ffn': ffn, 'ffn_qk': fq,
        'ffn_stats': ffn_stats, 'ffn_qk_stats': fq_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="3-way: Baseline / FFN-only / FFN+QK teleport"
    )
    p.add_argument('--smoke', action='store_true', help='Fast smoke test')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument(
        '--inner-steps', type=int, default=None, dest='inner_steps',
    )
    args = p.parse_args()

    if args.smoke:
        seeds = [0]
        steps = 20
        inner_steps = 5
        teleport_every = 5
    else:
        seeds = args.seeds if args.seeds is not None else [0, 1, 2]
        steps = args.steps if args.steps is not None else 100
        inner_steps = args.inner_steps if args.inner_steps is not None else 10
        teleport_every = 5

    batch_size = 16
    lr = 0.05

    print("=" * 60)
    print("3-Condition FFN+QK Benchmark")
    print(
        f"Seeds: {seeds}  Steps: {steps}"
        f"  teleport_every: {teleport_every}"
        f"  inner_steps: {inner_steps}"
    )
    print("=" * 60)

    all_results = []
    for seed in seeds:
        print(f"\nSeed {seed}:")
        res = _run_seed(seed, steps, inner_steps, teleport_every, batch_size, lr)
        all_results.append(res)

        base_auc = float(np.trapz(res['base']))
        ffn_auc = float(np.trapz(res['ffn']))
        fq_auc = float(np.trapz(res['ffn_qk']))
        s = res['ffn_stats']
        sq = res['ffn_qk_stats']

        print(
            f"  Final — base: {res['base'][-1]:.4f}"
            f"  ffn: {res['ffn'][-1]:.4f}"
            f"  ffn+qk: {res['ffn_qk'][-1]:.4f}"
        )
        print(
            f"  AUC   — base: {base_auc:.2f}"
            f"  ffn: {ffn_auc:.2f}"
            f"  ffn+qk: {fq_auc:.2f}"
        )
        print(
            f"  FFN    teleport: {s['accepted_count']}/{s['total_attempts']}"
            f" ({s['acceptance_rate']:.1f}%)"
        )
        print(
            f"  FFN+QK teleport: {sq['accepted_count']}/{sq['total_attempts']}"
            f" ({sq['acceptance_rate']:.1f}%)"
        )

    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Aggregate (mean across seeds)")
        print("=" * 60)
        stride = max(1, steps // 10)
        print(f"{'step':>5}  {'baseline':>10}  {'ffn':>10}  {'ffn+qk':>10}")
        for s in range(0, steps, stride):
            bv = [r['base'][s] for r in all_results]
            fv = [r['ffn'][s] for r in all_results]
            qv = [r['ffn_qk'][s] for r in all_results]
            print(
                f"{s:>5}  {np.mean(bv):>10.4f}"
                f"  {np.mean(fv):>10.4f}"
                f"  {np.mean(qv):>10.4f}"
            )

        base_aucs = [float(np.trapz(r['base'])) for r in all_results]
        ffn_aucs = [float(np.trapz(r['ffn'])) for r in all_results]
        fq_aucs = [float(np.trapz(r['ffn_qk'])) for r in all_results]
        print()
        print(
            f"Mean AUC  base: {np.mean(base_aucs):.2f}"
            f"  ffn: {np.mean(ffn_aucs):.2f}"
            f"  ffn+qk: {np.mean(fq_aucs):.2f}"
        )
        print(
            f"ΔAUC ffn vs base:    {np.mean(ffn_aucs)-np.mean(base_aucs):+.2f}"
        )
        print(
            f"ΔAUC ffn+qk vs base: {np.mean(fq_aucs)-np.mean(base_aucs):+.2f}"
        )
        print(
            f"ΔAUC ffn+qk vs ffn:  {np.mean(fq_aucs)-np.mean(ffn_aucs):+.2f}"
        )


if __name__ == '__main__':
    main()

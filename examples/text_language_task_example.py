#!/usr/bin/env python3
"""
Real text language task: character-level next-token prediction.

Demonstrates TeleportSGD on a real language modeling task using bundled
public-domain text snippets. No tokenizer or external data required.

Available corpora (--corpus flag):
    alice       Alice's Adventures in Wonderland excerpt (default)
    shakespeare Hamlet "To be, or not to be" soliloquy excerpt

Smoke test (fast, for CI):
    python examples/text_language_task_example.py --smoke

Full paired run:
    python examples/text_language_task_example.py
    python examples/text_language_task_example.py --corpus shakespeare
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from symmetry_teleport import TeleportSGD  # noqa: E402

# ---------------------------------------------------------------------------
# Bundled text corpora (public domain)
# ---------------------------------------------------------------------------

# Alice's Adventures in Wonderland (Lewis Carroll, 1865)
_CORPUS_ALICE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it, and what is "
    "the use of a book, thought Alice, without pictures or conversations? So she was "
    "considering in her own mind, as well as she could, for the hot day made her feel "
    "very sleepy and stupid, whether the pleasure of making a daisy-chain would be "
    "worth the trouble of getting up and picking the daisies, when suddenly a White "
    "Rabbit with pink eyes ran close by her. There was nothing so very remarkable in "
    "that; nor did Alice think it so very much out of the way to hear the Rabbit say "
    "to itself, Oh dear! Oh dear! I shall be late!"
)

# Hamlet, Act III Scene I (William Shakespeare, c. 1600)
_CORPUS_SHAKESPEARE = (
    "To be, or not to be, that is the question: "
    "Whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles "
    "and by opposing end them. To die, to sleep, "
    "no more; and by a sleep to say we end "
    "the heartache and the thousand natural shocks "
    "that flesh is heir to: tis a consummation "
    "devoutly to be wished. To die, to sleep; "
    "to sleep, perchance to dream. Ay, there's the rub, "
    "for in that sleep of death what dreams may come "
    "when we have shuffled off this mortal coil "
    "must give us pause. There's the respect "
    "that makes calamity of so long life."
)

_CORPORA = {
    "alice": _CORPUS_ALICE,
    "shakespeare": _CORPUS_SHAKESPEARE,
}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_sequences(text: str, stoi: dict, seq_len: int):
    """Sliding window over text: X[i]=tokens[i:i+T], Y[i]=tokens[i+1:i+T+1]."""
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = len(tokens) - seq_len
    X = torch.stack([tokens[i : i + seq_len] for i in range(n)])
    Y = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in range(n)])
    return X, Y


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------

class LMCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss that flattens (B, T, V) logits and (B, T) targets."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        V = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharLMTransformer(nn.Module):
    """
    Tiny character-level language model.

    Causal: each position attends only to itself and earlier positions via an
    upper-triangular additive mask. Token order is modeled by learned position
    embeddings added to character embeddings before the encoder.

    Implements get_ffn_layers(layer_idx) so it is compatible with TeleportSGD.
    The FFN uses ReLU activation, making diagonal scaling loss-invariant.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        nhead: int = 2,
        dim_feedforward: int = 64,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) integer tokens → (B, T, vocab_size) logits."""
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(positions)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        return self.lm_head(self.encoder(h, mask=causal_mask))

    def get_ffn_layers(self, layer_idx: int = 0):
        """Return (linear1, linear2) of the FFN block at layer_idx."""
        enc = self.encoder.layers[layer_idx]
        return enc.linear1, enc.linear2

    def get_attn_layer(self, layer_idx: int = 0):
        """Return the nn.MultiheadAttention module at layer_idx."""
        return self.encoder.layers[layer_idx].self_attn


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _minibatch_indices(step: int, batch_size: int, n: int) -> torch.Tensor:
    return torch.arange(step * batch_size, (step + 1) * batch_size) % n


def run_baseline(model, X, Y, loss_fn, lr, steps, batch_size):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = _minibatch_indices(step, batch_size, len(X))
        opt.zero_grad()
        loss = loss_fn(model(X[idx]), Y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def run_teleport(model, X, Y, loss_fn, lr, steps, batch_size, teleport_every, inner_steps):
    opt = TeleportSGD(
        model.parameters(),
        lr=lr,
        teleport_every=teleport_every,
        teleport_config={
            "model": model,
            "layer_idx": 0,
            "X_teleport": X,
            "Y_teleport": Y,
            "loss_fn": loss_fn,
            "lr_theta": 0.3,
            "inner_steps": inner_steps,
            "objective": "virtual_sgd_improve",
            "s_param": "projected",
            "log_s_clip": (-2.0, 2.0),
        },
    )
    losses = []
    for step in range(steps):
        idx = _minibatch_indices(step, batch_size, len(X))
        opt.zero_grad()
        loss = loss_fn(model(X[idx]), Y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses, opt.teleport_stats()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Char-level LM: baseline vs TeleportSGD")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (few steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps")
    parser.add_argument(
        "--corpus",
        choices=list(_CORPORA.keys()),
        default="alice",
        help="Text corpus to use (default: alice)",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=None,
        dest="inner_steps",
        help="Override inner optimization steps per teleport attempt",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    text = _CORPORA[args.corpus]
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    seq_len = 32
    X, Y = make_sequences(text, stoi, seq_len)

    lr = 0.05
    batch_size = 16
    if args.smoke:
        steps = 30
        inner_steps = 10
    elif args.steps is not None:
        steps = args.steps
        inner_steps = 20
    else:
        steps = 200
        inner_steps = 30
    if args.inner_steps is not None:
        inner_steps = args.inner_steps
    teleport_every = 5

    loss_fn = LMCrossEntropyLoss()

    print(f"Corpus      : {args.corpus}")
    print(f"Text length : {len(text)} chars")
    print(f"Vocab size  : {vocab_size}")
    print(f"Sequences   : {len(X)}")
    print(
        f"Steps       : {steps}  teleport_every={teleport_every}"
        f"  inner_steps={inner_steps}"
    )
    print()

    # Paired runs: identical init
    torch.manual_seed(args.seed)
    model_base = CharLMTransformer(
        vocab_size, d_model=32, nhead=2, dim_feedforward=64, max_seq_len=seq_len
    )
    shared_state = {k: v.clone() for k, v in model_base.state_dict().items()}

    model_tp = CharLMTransformer(
        vocab_size, d_model=32, nhead=2, dim_feedforward=64, max_seq_len=seq_len
    )
    model_tp.load_state_dict(shared_state)

    print("Running baseline SGD...")
    base_losses = run_baseline(model_base, X, Y, loss_fn, lr, steps, batch_size)

    print("Running TeleportSGD...")
    tp_losses, stats = run_teleport(
        model_tp, X, Y, loss_fn, lr, steps, batch_size, teleport_every, inner_steps
    )

    base_auc = float(np.trapz(base_losses))
    tp_auc = float(np.trapz(tp_losses))

    # Print table at 10 evenly-spaced checkpoints
    stride = max(1, steps // 10)
    print()
    print(f"{'step':>5}  {'baseline':>10}  {'teleport':>10}")
    for s in range(0, steps, stride):
        print(f"{s:>5}  {base_losses[s]:>10.4f}  {tp_losses[s]:>10.4f}")

    print()
    print(f"Final loss (baseline) : {base_losses[-1]:.4f}")
    print(f"Final loss (teleport) : {tp_losses[-1]:.4f}")
    print(f"Baseline AUC : {base_auc:.2f}")
    print(f"Teleport AUC : {tp_auc:.2f}")
    print(f"Delta AUC    : {tp_auc - base_auc:+.2f}")
    print()
    print(f"Teleport attempts : {stats['total_attempts']}")
    print(f"Accepted          : {stats['accepted_count']}"
          f" ({stats['acceptance_rate']:.1f}%)")


if __name__ == "__main__":
    main()

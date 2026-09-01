"""
demo/charlm.py — a promptable char-level model trained the consolidation way.

Wraps `ConsolidatingNet` (experiments/consolidation_rounds.py) as a fixed-context
next-char predictor. Input = a window of W one-hot chars -> [W*V] vector; output =
V next-char logits. Reuses the frozen spine + branch pattern (multi_teacher) so we
can ADD a second corpus as a frozen branch with ZERO forgetting of the first, and
`forward_logits` (society) so a frozen net can be evaluated/generated on NEW contexts.

This is a TINY, fixed-context, char-level model — NOT a large language model, NOT
fluent. It learns local next-char structure. The claim is about the TRAINING METHOD
(no-forgetting, interpretable least-path traces, capability-per-synapse), not scale.

Pure numpy, CPU. No network, no sklearn, no torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_feats, forward_logits
from experiments.synaptic_pruning import MLP, train


# ---------------------------------------------------------------------------
# corpus / vocab / windowing
# ---------------------------------------------------------------------------
def load_corpus(path):
    """Read a corpus file, dropping the '=== NAME ===' section markers."""
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith("===") or line.strip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_vocab(texts):
    """Sorted unique characters across all corpora (stable, deterministic)."""
    return sorted(set("".join(texts)))


def window(text, W, vocab):
    """Slide a window of W chars over `text`; return (X one-hot [N, W*V], y next-char [N]).

    Each row is the one-hot of the W chars ending at position i; y[i] is the char
    that follows. Only positions with a full window are used.
    """
    V = len(vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    n = max(0, len(text) - W)
    X = np.zeros((n, W * V), dtype=float)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        for k in range(W):
            c = text[i + k]
            if c in idx:
                X[i, k * V + idx[c]] = 1.0
        nxt = text[i + W]
        y[i] = idx.get(nxt, 0)
    return X, y


def split_windows(X, y, frac=0.8, seed=0):
    """Deterministic train/test split of windowed data."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    ntr = int(frac * len(y))
    return X[perm[:ntr]], y[perm[:ntr]], X[perm[ntr:]], y[perm[ntr:]]


# ---------------------------------------------------------------------------
# training (consolidation) + the traditional MLP control
# ---------------------------------------------------------------------------
def train_charlm(Xtr, ytr, Xte, yte, D, C, seed=1, rounds=2, P=32, EP=600, tau_ramp=0.6):
    """A standalone char-LM spine: consolidation rounds on one corpus.

    tau_ramp is the tightening ratio ceiling; round 1 is pure axioms (tau=0), later
    rounds ramp up to pull fibers into cross-paths (the architecture's signature).
    """
    net = ConsolidatingNet(D, C, seed=seed)
    for r in range(rounds):
        tau = 0.0 if r == 0 else min(tau_ramp, 0.3 + 0.15 * r)
        net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=tau,
                       prune_density=(0.8 if r else None))
    return net


def spine_features_on(spine, X):
    """Recompute a frozen spine's features on ARBITRARY inputs X (any length)."""
    As = forward_feats(spine, X)
    return np.concatenate(As, 1) if As else np.zeros((len(X), 0))


def grow_branch_on(spine, Xtr, ytr, Xte, yte, D, C, seed=2, P=32, EP=600):
    """A per-corpus branch off a frozen spine, on data the spine has NOT seen."""
    br = ConsolidatingNet(D, C, seed=seed)
    br.seed_base(spine_features_on(spine, Xtr), spine_features_on(spine, Xte),
                 list(spine.dist), [w.copy() for w in spine.frozen_W])
    br.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=0.6, k_par=6, prune_density=0.7)
    return br


def train_mlp(Xtr, ytr, Xte, yte, D, C, H=256, epochs=1500, lr=0.05, wd=1e-4, seed=3):
    """The traditional control: a plain MLP (gradient descent, no freezing)."""
    net = MLP(D, H, C, seed=seed)
    train(net, Xtr, ytr, Xte, yte, epochs, lr, wd)
    return net


# ---------------------------------------------------------------------------
# evaluation + generation
# ---------------------------------------------------------------------------
def next_char_acc(net, X, y):
    """Fraction of next-char predictions correct on (X, y)."""
    return float((forward_logits(net, X).argmax(1) == y).mean())


def bits_per_char(net, X, y):
    """Mean negative log2 probability of the true next char (cross-entropy in bits)."""
    logits = forward_logits(net, X)
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(1, keepdims=True)
    p = np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)
    return float(-np.log2(p).mean())


def generate(net, prompt, n, temp, vocab, W):
    """Autoregressive continuation: sample n chars after `prompt` at temperature `temp`."""
    V = len(vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    ctx = list(prompt)
    out = []
    for _ in range(n):
        tail = ctx[-W:]
        x = np.zeros((1, W * V), dtype=float)
        for k, c in enumerate(tail):
            if c in idx:
                x[0, k * V + idx[c]] = 1.0
        logits = forward_logits(net, x)[0]
        logits = logits / max(temp, 1e-6)
        logits -= logits.max()
        e = np.exp(logits)
        p = e / e.sum()
        c = vocab[int(np.random.choice(V, p=p))]
        out.append(c)
        ctx.append(c)
    return "".join(out)


# ---------------------------------------------------------------------------
# export for the web artifact
# ---------------------------------------------------------------------------
def export_json(net, vocab, W, path):
    """Serialize a char-LM so a JS port can reimplement forward_logits + sampling."""
    data = {
        "window": W,
        "vocab": list(vocab),
        "bias": net.bias.tolist(),
        "rounds": [
            {"W": w.tolist(), "V": v.tolist(), "b": b.tolist()}
            for w, v, b in zip(net.frozen_W, net.frozen_V, net.frozen_b)
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data))
    return path

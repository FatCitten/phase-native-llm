"""
demo/foster.py — the foster-parent harness: a frontier LLM raises the Nautilus child.

Two mechanisms, both real:
  1. DISTILLATION — the teacher LLM provides SOFT TARGETS (probability distributions
     over the next word) for a sample of contexts, and the child trains to match those
     instead of hard one-hot labels. The child inherits the teacher's knowledge.
     Implemented in the core: ConsolidatingNet.grow_round(soft=...) trains against the
     teacher's distribution (e.g. that 'the' and 'a' are both plausible next words).
  2. STRUCTURAL ENGINEERING — the teacher gets more tools through the harness to
     surgically shape the child's architecture (grow branches, add capacity).

This module implements mechanism 1 (distillation). It:
  - asks the teacher (an LLM via Ollama Cloud) for a next-word distribution given a
    context of W words
  - trains the child (ConsolidatingNet) to match those soft targets via grow_round(soft=)
  - measures the improvement over hard-label training

Honest framing: the child is tiny and on a laptop. It will be competitive FOR ITS SIZE,
not match the teacher. But distillation is the proven path to a competitive small model.

Pure numpy + stdlib. Run: python -m demo.foster --model glm-5.3-flash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import wordlm
from demo.demo import load_sections
from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits
from phase_native.ollama_agent import OllamaClient

DEMO = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# teacher soft targets
# ---------------------------------------------------------------------------
def teacher_next_word_dist(client, model, context_words, vocab, top_k=20):
    """Ask the teacher for a probability distribution over the next word.

    Returns a dict {word: prob} over the vocab (only top_k words get nonzero prob;
    the rest are folded into <UNK>). The teacher is prompted to give the most likely
    next word and a few alternatives with probabilities.
    """
    ctx = " ".join(context_words)
    prompt = (
        f"Given the text context: \"{ctx}\"\n"
        f"Predict the NEXT WORD. Return ONLY a JSON object mapping the top {top_k} "
        f"most likely next words to their probabilities (0..1, summing to 1). "
        f"Use lowercase words only. Example: {{\"the\": 0.4, \"and\": 0.3, \"was\": 0.3}}"
    )
    resp = client.chat(model, [
        {"role": "system", "content": "You are a language model predicting the next word. "
         "Return ONLY valid JSON, no prose."},
        {"role": "user", "content": prompt},
    ], [])
    text = resp["choices"][0]["message"].get("content", "")
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        dist = json.loads(text[start:end])
    except Exception:
        return None
    # normalize + fold into vocab
    out = {}
    total = 0.0
    for w, p in dist.items():
        w = w.lower().strip()
        if w in vocab and p > 0:
            out[w] = float(p)
            total += float(p)
    if not out or total <= 0:
        return None
    for w in out:
        out[w] /= total
    return out


def soft_targets_for_contexts(client, model, contexts, vocab, top_k=20, delay=0.2):
    """Get soft targets for a list of contexts. Returns list of (context, dist|None)."""
    results = []
    for ctx in contexts:
        d = teacher_next_word_dist(client, model, ctx, vocab, top_k=top_k)
        results.append((ctx, d))
        time.sleep(delay)  # be polite to the API
    return results


# ---------------------------------------------------------------------------
# distillation training
# ---------------------------------------------------------------------------
def soft_target_matrix(contexts, dists, vocab, W):
    """Build X (one-hot contexts) and a soft-target matrix P (N x C) from teacher dists.

    For contexts where the teacher returned None, fall back to a uniform target.
    Returns (X, P, valid_mask).
    """
    V = len(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(contexts), W * V), dtype=float)
    P = np.zeros((len(contexts), V), dtype=float)
    valid = np.zeros(len(contexts), dtype=bool)
    for i, (ctx, dist) in enumerate(zip(contexts, dists)):
        for k, w in enumerate(ctx):
            X[i, k * V + idx.get(w, 0)] = 1.0
        if dist:
            for w, p in dist.items():
                P[i, idx.get(w, 0)] = p
            valid[i] = True
        else:
            P[i] = 1.0 / V  # uniform fallback
    return X, P, valid


def train_distill(net, X, P, valid, Xtr, ytr, Xte, yte, rounds=2, P_fib=48, epochs=200):
    """Train the child to match teacher soft targets via grow_round(soft=...).

    The child grows consolidation rounds where the training target is the teacher's
    soft distribution (not hard one-hot labels). This is the real distillation path:
    the child's structure is shaped by the teacher's knowledge.
    """
    # only use valid (teacher-answered) rows for distillation
    Xv, Pv = X[valid], P[valid]
    yv = ytr[:len(Xv)] if len(Xv) <= len(ytr) else None
    # grow rounds against soft targets
    for r in range(rounds):
        net.grow_round(Xv, yv, Xte, yte, P=P_fib, epochs=epochs, tau=0.0,
                       floor=0.05, soft=Pv)
    return net


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--n-contexts", type=int, default=50, help="contexts to distill")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    print("=" * 72)
    print("FOSTER PARENT — a frontier LLM raises the Nautilus child (distillation)")
    print("=" * 72)
    print(f"teacher: {args.model}  contexts: {args.n_contexts}  top-k: {args.top_k}")

    # data
    sections = load_sections(DEMO / "corpus.txt")
    texts = list(sections.values())
    vocab = wordlm.build_vocab(texts, min_freq=5)
    W = 4
    tokens = wordlm.tokenize(" ".join(texts))
    X, y = wordlm.window_words(tokens, W, vocab)
    Xtr, ytr, Xte, yte = wordlm.split(X, y, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)
    print(f"vocab={len(vocab)} D={D} C={C} train={len(ytr)} test={len(yte)}")

    # baseline child (hard labels)
    print("\n--- training baseline child (hard labels) ---")
    net0 = ConsolidatingNet(D, C, seed=1)
    for r in range(3):
        net0.grow_round(Xtr, ytr, Xte, yte, P=48, epochs=200, tau=0.0, floor=0.05)
    acc0 = float((forward_logits(net0, Xte).argmax(1) == yte).mean())
    print(f"baseline child acc: {acc0:.4f}")

    # teacher soft targets
    print(f"\n--- asking teacher for soft targets on {args.n_contexts} contexts ---")
    client = OllamaClient(host="http://localhost:11434" if args.local else None, timeout=120)
    # sample contexts from the training set
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(ytr), size=min(args.n_contexts, len(ytr)), replace=False)
    contexts = []
    for i in sample_idx:
        ctx = []
        for k in range(W):
            onehot = Xtr[i, k*C:(k+1)*C]
            w = vocab[int(onehot.argmax())]
            ctx.append(w)
        contexts.append(ctx)

    t0 = time.time()
    results = soft_targets_for_contexts(client, args.model, contexts, vocab, top_k=args.top_k)
    dt = time.time() - t0
    n_valid = sum(1 for _, d in results if d)
    print(f"got soft targets for {n_valid}/{len(results)} contexts in {dt:.0f}s")

    # build soft-target matrix
    Xs, Ps, valid = soft_target_matrix(contexts, results, vocab, W)
    print(f"soft-target matrix: {Xs.shape}, {valid.sum()} valid rows")

    # distill into a fresh child
    print(f"\n--- distilling teacher knowledge into a new child ({args.rounds} rounds, {args.epochs} ep) ---")
    net1 = ConsolidatingNet(D, C, seed=1)
    train_distill(net1, Xs, Ps, valid, Xtr, ytr, Xte, yte,
                  rounds=args.rounds, P_fib=48, epochs=args.epochs)
    acc1 = float((forward_logits(net1, Xte).argmax(1) == yte).mean())
    print(f"distilled child acc: {acc1:.4f}  (baseline was {acc0:.4f})")

    print("\n" + "=" * 72)
    print(f"RESULT: baseline {acc0:.4f} -> distilled {acc1:.4f}  (Δ {acc1-acc0:+.4f})")
    print("=" * 72)


if __name__ == "__main__":
    main()

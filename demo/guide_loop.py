"""demo/guide_loop.py — the LLM guides, the structure decides. The headline loop.

The teacher (a frontier LLM) emits twinges: soft next-word distributions on a batch
of contexts. The child grows a consolidation round with soft=... — its own
overproduce->prune->freeze dynamics decide which signals STICK. We measure
capability-per-synapse + no-forgetting after each round.

Honest framing: the teacher GUIDES, the child's structure DECIDES. If the teacher's
twinges don't improve the metrics, we report it plainly — the mechanism needs tuning,
not a forced win.

Run: python -m demo.guide_loop --model glm-5.3-flash --rounds 3 --contexts 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import signal_engine, wordlm
from demo.demo import load_sections
from experiments.consolidation_rounds import ConsolidatingNet
from phase_native.ollama_agent import OllamaClient

DEMO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--contexts", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("THE LLM GUIDES, THE STRUCTURE DECIDES")
    print("=" * 72)
    print(f"teacher: {args.model}  rounds: {args.rounds}  contexts: {args.contexts}")

    # data — small vocab, subsampled, so the loop is FAST (fast or useless)
    sections = load_sections(DEMO / "corpus.txt")
    vocab = wordlm.build_vocab(list(sections.values()), min_freq=20)
    W = 3
    tokens = wordlm.tokenize(" ".join(sections.values()))
    X, y = wordlm.window_words(tokens, W, vocab)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), size=min(2000, len(y)), replace=False)
    X, y = X[keep], y[keep]
    Xtr, ytr, Xte, yte = wordlm.split(X, y, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)
    print(f"vocab={len(vocab)} D={D} C={C} train={len(ytr)} test={len(yte)}")

    # baseline child (hard labels)
    net = ConsolidatingNet(D, C, seed=1)
    for r in range(2):
        net.grow_round(Xtr, ytr, Xte, yte, P=24, epochs=args.epochs, tau=0.0)
    se = signal_engine.SignalEngine(net, vocab, W, Xte, yte, Xte, yte)
    print("\nbaseline:", json.dumps(se.measure(), default=float))

    client = OllamaClient(host="http://localhost:11434" if args.local else None, timeout=120)

    for r in range(args.rounds):
        # sample contexts, ask the teacher for twinges
        rng = np.random.default_rng(r)
        idx = rng.choice(len(Xtr), size=min(args.contexts, len(Xtr)), replace=False)
        contexts = [[vocab[int(Xtr[i, k])] for k in range(W)] for i in idx]
        print(f"\n--- round {r+1}: asking teacher for twinges on {len(contexts)} contexts ---")
        t0 = time.time()
        twinges = se.teacher_twinge(client, args.model, contexts)
        n_valid = sum(1 for _, d in twinges if d)
        print(f"got {n_valid}/{len(twinges)} twinges in {time.time()-t0:.0f}s")

        # build soft-target matrix from the twinges
        from demo.foster import soft_target_matrix
        Xs, Ps, valid = soft_target_matrix(contexts, [d for _, d in twinges], vocab, W)

        # grow the child on the teacher's signals (the child decides what sticks)
        stats = se.grow_on_signals(Xs, np.zeros(len(Xs), dtype=int), P=24,
                                   epochs=args.epochs, tau=0.0, soft=Ps)
        print(f"grew round: kept {stats['kept']}/24, void_frac={stats['void_frac']:.2f}")
        print(f"after round {r+1}: {json.dumps(se.measure(), default=float)}")

    # save the grown child
    from demo.engine import StructureEngine
    StructureEngine(net).save_structure("results/child_grown.json")
    print("\nsaved results/child_grown.json")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
demo/superiority.py — does consolidation actually win on real text?

Sweeps consolidation configs (window W, rounds, candidate pool P) on the ALICE
corpus and pits the consolidation char-LM (train_charlm) against a matched
traditional MLP (train_mlp, H=256, epochs=rounds*1000) on held-out next-char
accuracy. Honest comparison: the MLP is given the SAME epoch budget as the
consolidation loop, so any win (or loss) is a real property of the method, not
of compute.

Consolidation configs: W in {8,16}, rounds in {3,6}, P in {64,96} -> 8 configs.
The MLP control depends only on `rounds`, so it is cached per rounds value and
reused across W/P (identical net, identical data) — no free lunch, just no
double-training of an identical model.

Pure numpy, CPU. Run: cd /tmp/phase-native-llm && source .venv/bin/activate
                        && python -m demo.superiority
Saves results/superiority.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo.charlm import (build_vocab, window, split_windows,
                          train_charlm, next_char_acc, train_mlp)
from demo.demo import load_sections

RESULTS = Path("results")


def main():
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    # ---- corpus: ALICE only, deterministic seed so W/P/rounds all share data
    sects = load_sections(Path(__file__).resolve().parent / "corpus.txt")
    text = sects["ALICE"]
    vocab = build_vocab([text])
    V = len(vocab)
    SEED = 7

    # ---- sweep configs
    configs = [(W, rounds, P)
               for W in (8, 16)
               for rounds in (3, 6)
               for P in (64, 96)]

    print(f"ALICE corpus: {len(text)} chars, vocab={V}, {len(configs)} configs, "
          f"seed={SEED}")

    rows = []          # list of dict rows
    mlp_cache = {}     # rounds -> mlp net (independent of W/P)

    for W, rounds, P in configs:
        D = W * V
        rng = np.random.default_rng(SEED)
        # window over 600 chars, shuffle, deterministic train/test split
        Xs, ys = window(text, W, vocab)
        n = min(600, len(ys))
        perm = rng.permutation(len(ys))[:n]
        X, y = Xs[perm], ys[perm]
        ntr = int(0.8 * n)
        Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

        # ---- consolidation char-LM (its own training data split, but same rows
        #      since window() used rng-independent slicing above would differ;
        #      we re-windowed directly so reuse Xtr/ytr/Xte/yte for both)
        t = time.time()
        cnet = train_charlm(Xtr, ytr, Xte, yte, D, V,
                            seed=11, rounds=rounds, P=P)
        cons_acc = next_char_acc(cnet, Xte, yte)
        cons_syn = cnet.synapses
        t_cons = time.time() - t

        # ---- matched MLP: cached by (rounds, W) — identical net & data (input
        #      dim D=W*V differs across W, so W must be part of the key)
        key = (rounds, W)
        if key not in mlp_cache:
            t = time.time()
            mnet = train_mlp(Xtr, ytr, Xte, yte, D, V, H=256,
                             epochs=rounds * 1000, seed=13)
            mlp_cache[key] = (mnet, time.time() - t)
        mnet, t_mlp = mlp_cache[key]
        mlp_acc = mnet.acc(Xte, yte)

        delta = cons_acc - mlp_acc
        rows.append(dict(W=W, rounds=rounds, P=P,
                         cons_acc=round(float(cons_acc), 4),
                         mlp_acc=round(float(mlp_acc), 4),
                         delta=round(float(delta), 4),
                         cons_syn=int(cons_syn),
                         t_cons=round(t_cons, 1), t_mlp=round(t_mlp, 1)))
        print(f"  W={W:2d} rounds={rounds} P={P:3d}  cons={cons_acc:.4f}  "
              f"mlp={mlp_acc:.4f}  delta={delta:+.4f}  "
              f"({t_cons:.1f}s / {t_mlp:.1f}s)", flush=True)

    # ---- winners + capability-per-synapse at the best config
    best = max(rows, key=lambda r: r["delta"])
    best_cons_syn = best["cons_syn"]
    # capability-per-synapse: accuracy per synapse (x1000), for the best-delta config
    cps_cons = best["cons_acc"] / best_cons_syn * 1e3
    # MLP matched synapse count at that config: H*(D+C)
    Dbest = best["W"] * V
    cps_mlp = best["mlp_acc"] / (256 * (Dbest + V)) * 1e3

    wins = sum(1 for r in rows if r["delta"] > 0)
    total = len(rows)

    # ---- final table (fixed ordering: W, rounds, P)
    print("\n  W  rounds  P    cons      mlp      delta")
    print("  " + "-" * 40)
    for W in (8, 16):
        for rounds in (3, 6):
            for P in (64, 96):
                r = next(x for x in rows if (x["W"], x["rounds"], x["P"]) == (W, rounds, P))
                print(f"  {W:2d}  {rounds:5d}  {P:3d}  {r['cons_acc']:.4f}  "
                      f"{r['mlp_acc']:.4f}  {r['delta']:+.4f}")

    verdict = (f"consolidation won {wins}/{total} configs "
               f"({'BEATS' if wins > total / 2 else 'trails'} matched MLP).")
    print(f"\nBest delta: W={best['W']} rounds={best['rounds']} P={best['P']} "
          f"cons={best['cons_acc']:.4f} vs mlp={best['mlp_acc']:.4f} "
          f"(delta {best['delta']:+.4f})")
    print(f"Capability/synapse at best config: cons={cps_cons:.4f}e-3  "
          f"mlp={cps_mlp:.4f}e-3")
    print(f"VERDICT: {verdict}")

    res = {
        "corpus": "ALICE", "vocab_size": V, "seed": SEED,
        "wins_count": wins, "total_configs": total,
        "best_config": {k: best[k] for k in ("W", "rounds", "P", "cons_acc", "mlp_acc", "delta")},
        "best_config_delta": best["delta"],
        "capability_per_synapse": {
            "consolidation": round(float(cps_cons), 6),
            "mlp": round(float(cps_mlp), 6),
        },
        "configs": rows,
        "runtime_sec": round(time.time() - t0, 1),
        "verdict": verdict,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "superiority.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {RESULTS / 'superiority.json'} in {res['runtime_sec']}s")


if __name__ == "__main__":
    main()

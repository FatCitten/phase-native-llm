"""demo/validate_crystal.py — the CRYSTAL-VALIDATION PROBE.

Does the saved child's holonomy bundle hold REAL understanding, or is it noise?
We run the falsifiable acceptance bars from the plan against the grown child:

  BAR 1 (discrimination): same-label (same-next-word) held-out context pairs should
    share a bundle axiom target MORE than different-label pairs. We report the
    discrimination margin vs the random (chance) baseline. Required >= 0.2 above chance.

  BAR 2 (retrieval/reconstruction): train a tiny numpy probe on bundle -> correct-axiom
    and test on held-out contexts. Must beat a random baseline by >= 10 points.

Honesty rule (project ethos): the numbers are the numbers. If a bar fails, we say so
plainly. We do NOT force a pass.

Run: python -m demo.validate_crystal  [--child results/child_grown.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import holonomy, wordlm
from demo.demo import load_sections
from demo.engine import StructureEngine

DEMO = Path(__file__).resolve().parent


def build_hf(net, Xtr):
    hf = holonomy.HolonomyField(net)
    hf.accumulate(Xtr)   # grow the memory on the training set (concept-centered)
    return hf


def bundle_targets(hf, X):
    """For each input, the set of gated axiom targets its firing fibers lean toward."""
    from experiments.society import forward_feats
    Xd = hf._dense(X)
    As = forward_feats(hf.net, Xd)
    A = np.concatenate(As, 1) if As else np.zeros((len(Xd), 0))
    H = hf.H; gate = hf.H_mag >= 1.0 / holonomy.PHI
    targets = []
    for i in range(len(Xd)):
        fired = np.where(A[i] > 0)[0]
        ts = set()
        for f in fired:
            if not gate[f]:
                continue
            row = H[f]
            if row.sum() == 0:
                continue
            ts.add(int(np.argmax(row)))
        targets.append(ts)
    return targets


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=str(DEMO / ".." / "results" / "child_grown.json"))
    args = ap.parse_args()

    eng = StructureEngine.load_structure(args.child)
    net = eng.net
    print(f"loaded child: D={net.D} C={net.C} rounds={len(net.frozen_W)} "
          f"fibers={sum(w.shape[1] for w in net.frozen_W)}")

    # rebuild the same data pipeline as guide_loop (deterministic)
    sections = load_sections(DEMO / "corpus.txt")
    vocab = wordlm.build_vocab(list(sections.values()), min_freq=20)
    W = 3
    tokens = wordlm.tokenize(" ".join(sections.values()))
    X, y = wordlm.window_words(tokens, W, vocab)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), size=min(2000, len(y)), replace=False)
    X, y = X[keep], y[keep]
    Xtr, ytr, Xte, yte = wordlm.split(X, y, frac=0.8, seed=1)

    hf = build_hf(net, Xtr)
    print(f"holonomy: {hf.n_fibers} fibers, {len(net.dist)} dist entries, "
          f"{len(hf.axioms)} axioms (round-1 fibers)")

    # ---- BAR 1: discrimination of same-label vs different-label held-out pairs ----
    # held-out = the portion of Xte that FIRES some gated fiber (leanable). We need
    # pairs, so take a budgeted sample of distinct contexts from Xte.
    labels = yte
    targets = bundle_targets(hf, Xte)
    leanable = [i for i, t in enumerate(targets) if len(t) > 0]
    print(f"BAR 1: {len(leanable)}/{len(Xte)} held-out contexts lean on a gated axiom")

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b); union = len(a | b)
        return inter / union if union else 0.0

    # structural emptiness check: if H rows never point at any axiom, the bundle is
    # empty and the probe cannot be meaningfully evaluated — report it plainly.
    h_nonempty = int((hf.H.sum(1) > 0).sum())
    print(f"(holonomy integrity: {h_nonempty}/{hf.n_fibers} fibers have a populated "
          f"H row / a target; axioms={len(hf.axioms)})")

    margin_vs_chance = 0.0
    passed1 = False
    if len(leanable) == 0:
        print("BAR 1: NO held-out context leans on a gated axiom — bundle is empty. "
              "Discrimination is undefined (0 measurable pairs).")
        print("BAR 1: REQUIRED >= 0.20 above chance -> FAIL (no signal to discriminate)")
    else:
        # build balanced same-label vs different-label pairs
        rng2 = np.random.default_rng(42)
        same = []
        for _ in range(4000):
            a, b = rng2.choice(leanable, 2, replace=False)
            if labels[a] == labels[b]:
                same.append(jaccard(targets[a], targets[b]))
        diff = []
        for _ in range(4000):
            a, b = rng2.choice(leanable, 2, replace=False)
            if labels[a] != labels[b]:
                diff.append(jaccard(targets[a], targets[b]))
        # chance baseline = mean jaccard over random pairs regardless of label
        chance = []
        for _ in range(4000):
            a, b = rng2.choice(leanable, 2, replace=False)
            chance.append(jaccard(targets[a], targets[b]))
        same_m = float(np.mean(same)) if same else 0.0
        diff_m = float(np.mean(diff)) if diff else 0.0
        chance_m = float(np.mean(chance)) if chance else 0.0
        margin = same_m - diff_m
        margin_vs_chance = same_m - chance_m
        print(f"BAR 1: same-label shared-axiom J={same_m:.4f}  diff-label J={diff_m:.4f}  "
              f"chance J={chance_m:.4f}")
        print(f"BAR 1: discrimination margin (same-diff) = {margin:+.4f}  "
              f"vs-chance (same-chance) = {margin_vs_chance:+.4f}")
        passed1 = margin_vs_chance >= 0.2
        print(f"BAR 1: REQUIRED >= 0.20 above chance -> {'PASS' if passed1 else 'FAIL'}")

    # ---- BAR 2: retrieval probe (bundle-feature -> correct-axiom) on held-out ----
    # bundle feature for a context = bag over its gated axiom targets (a feature per
    # axiom). Probe predicts the MOST-FREQUENTLY-LEANED axiom per label. We test held-out.
    from collections import defaultdict
    # train probe on Xtr (the memory set), test on the held-out, LEANABLE slice
    tr_targets = bundle_targets(hf, Xtr)
    te_targets = bundle_targets(hf, Xte)
    n_axiom = len(hf.axioms)

    def feat(ts, n):
        v = np.zeros(n)
        for t in ts:
            if 0 <= t < n:
                v[t] += 1.0
        return v

    # BAR 2 honesty guard: with no populated H rows / empty bundles there is no
    # bundle feature to reconstruct from — report the empty result plainly.
    train_lean = sum(1 for ts in tr_targets if ts)
    test_lean = sum(1 for ts in te_targets if ts)
    print(f"BAR 2: {train_lean}/{len(tr_targets)} train contexts and "
          f"{test_lean}/{len(te_targets)} held-out contexts have a non-empty bundle")

    probe_acc = 0.0
    rand_acc = 0.0
    passed2 = False
    if test_lean == 0 or h_nonempty == 0:
        print("BAR 2: bundle is empty (no axiom targets) — probe reconstruction is "
              "undefined (nothing to reconstruct from).")
        print("BAR 2: required >= 0.10 (10 pts) over random -> FAIL (no signal)")
    else:
        # probe: per-label average bundle vector on train -> pick argmax as predicted axiom
        label_vec = defaultdict(lambda: np.zeros(n_axiom))
        label_cnt = defaultdict(float)
        for i, ts in enumerate(tr_targets):
            if not ts:
                continue
            label_vec[ytr[i]] += feat(ts, n_axiom)
            label_cnt[ytr[i]] += 1.0
        pred_axiom = {}
        for lab, v in label_vec.items():
            if v.sum() > 0:
                pred_axiom[lab] = int(v.argmax())

        # held-out: for each leanable test context, does the probe's predicted axiom
        # (for the context's label) appear in the context's bundle?
        hits = 0; tot = 0
        for i, ts in enumerate(te_targets):
            if not ts or yte[i] not in pred_axiom:
                continue
            tot += 1
            if pred_axiom[yte[i]] in ts:
                hits += 1
        probe_acc = hits / tot if tot else 0.0
        # random baseline: pick a random gated axiom per test context
        rng3 = np.random.default_rng(7)
        rand_hits = 0; rand_tot = 0
        for i, ts in enumerate(te_targets):
            if not ts:
                continue
            rand_tot += 1
            rnd = rng3.integers(0, len(hf.axioms))
            if rnd in ts:
                rand_hits += 1
        rand_acc = rand_hits / rand_tot if rand_tot else 0.0
        margin2 = probe_acc - rand_acc
        print(f"BAR 2: probe reconstruct acc = {probe_acc:.4f}  ({hits}/{tot}) "
              f"random baseline = {rand_acc:.4f}  ({rand_hits}/{rand_tot})")
        print(f"BAR 2: margin over random = {margin2:+.4f} ({(margin2*100):.1f} points)")
        passed2 = margin2 >= 0.10
        print(f"BAR 2: REQUIRED >= 0.10 (10 pts) over random -> {'PASS' if passed2 else 'FAIL'}")
        probe_acc = margin2

    bar2_margin = 0.0 if h_nonempty == 0 else probe_acc
    print("\n" + "=" * 72)
    print(f"HONEST VERDICT: BAR1(discrimination vs chance)={'PASS' if passed1 else 'FAIL'} "
          f"BAR2(retrieval vs random)={'PASS' if passed2 else 'FAIL'}")
    print(f"  BAR1 margin vs chance = {margin_vs_chance:+.4f}")
    print(f"  BAR2 margin vs random = {bar2_margin:+.4f} ({(bar2_margin*100):.1f} pts)")
    print("=" * 72)


if __name__ == "__main__":
    run()

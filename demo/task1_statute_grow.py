"""Nautilus Task 1 writer: grow the legal statute as ADDITIONAL ROUNDS on the SAME
spine net (built on ALICE), then compute the overlap metric (Jaccard of gated axiom
targets between ALICE held-out and STATUTE held-out contexts)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import holonomy, wordlm
from demo.demo import load_sections
from experiments.consolidation_rounds import ConsolidatingNet

DEMO = Path(__file__).resolve().parent

def build_data(section_text, vocab, W=3, seed=0, n=2000):
    tokens = wordlm.tokenize(section_text)
    X, y = wordlm.window_words(tokens, W, vocab)
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(y), size=min(n, len(y)), replace=False)
    X, y = X[keep], y[keep]
    Xtr, ytr, Xte, yte = wordlm.split(X, y, frac=0.8, seed=1)
    return Xtr, ytr, Xte, yte

def bundle_targets(hf, X):
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

def main():
    sections = load_sections(DEMO / "corpus.txt")
    vocab = wordlm.build_vocab(list(sections.values()), min_freq=20)
    W = 3
    D = W * len(vocab); C = len(vocab)
    print(f"vocab={len(vocab)} D={D} C={C}")

    # ---- build the SPINE on ALICE (rounds 1-2) ----
    alice_tr, alice_ytr, alice_te, alice_yte = build_data(sections["ALICE"], vocab)
    net = ConsolidatingNet(D, C, seed=1)
    for r in range(2):
        net.grow_round(alice_tr, alice_ytr, alice_te, alice_yte, P=24, epochs=100, tau=0.0)
    print(f"spine built on ALICE: rounds={len(net.frozen_W)} fibers={sum(w.shape[1] for w in net.frozen_W)}")

    # ---- grow the STATUTE as ADDITIONAL ROUNDS on the SAME spine ----
    stat_tr, stat_ytr, stat_te, stat_yte = build_data(sections["STATUTE"], vocab)
    for r in range(2):
        net.grow_round(stat_tr, stat_ytr, stat_te, stat_yte, P=24, epochs=100, tau=0.0)
    print(f"after statute rounds: rounds={len(net.frozen_W)} fibers={sum(w.shape[1] for w in net.frozen_W)}")

    # ---- holonomy field + overlap metric ----
    # accumulate memory on the combined training set (spine + statute)
    hf = holonomy.HolonomyField(net)
    hf.accumulate(np.concatenate([alice_tr, stat_tr], 0))
    print(f"holonomy: {hf.n_fibers} fibers, {len(hf.axioms)} axioms")

    a_targets = bundle_targets(hf, alice_te)
    n_targets = bundle_targets(hf, stat_te)

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b); union = len(a | b)
        return inter / union if union else 0.0

    # aggregate: union of all gated axiom targets per domain
    A_union = set().union(*[t for t in a_targets if t])
    N_union = set().union(*[t for t in n_targets if t])
    overlap = jaccard(A_union, N_union)
    print(f"ALICE held-out leanable: {sum(1 for t in a_targets if t)}/{len(a_targets)}")
    print(f"STATUTE held-out leanable: {sum(1 for t in n_targets if t)}/{len(n_targets)}")
    print(f"ALICE axiom-target union size: {len(A_union)}")
    print(f"STATUTE axiom-target union size: {len(N_union)}")
    print(f"OVERLAP (Jaccard of gated axiom targets) = {overlap:.4f}")

    # also per-context mean Jaccard between the two domains
    rng = np.random.default_rng(0)
    a_lean = [i for i, t in enumerate(a_targets) if t]
    n_lean = [i for i, t in enumerate(n_targets) if t]
    if a_lean and n_lean:
        js = [jaccard(a_targets[a], n_targets[b])
              for a in rng.choice(a_lean, min(500, len(a_lean)), replace=False)
              for b in rng.choice(n_lean, min(500, len(n_lean)), replace=False)]
        print(f"mean cross-domain per-context Jaccard = {np.mean(js):.4f}")

    return overlap

if __name__ == "__main__":
    main()

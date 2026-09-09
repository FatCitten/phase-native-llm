"""The HOLONOMY FIELD (memory) + LEAN BUNDLE (retrieval).

Spines as a crystallized understanding-retrieval source: the system does NOT
"understand" -- it uses HOLONOMY over CURVATURE to determine LEANS. Each fiber
accumulates a directional vector from past inputs ("what happened because of
the past") = the MEMORY/retrieval mechanism. The LEAN of a fiber = the
direction its accumulated holonomy points. A query perturbs the field; each
firing fiber leans toward the AXIOM (distance-0 fiber) its holonomy has
accumulated toward. Reaching the golden-ratio (PHI) threshold gates retention
AND retrieval. The final output = a VECTOR BUNDLE of routes plainly pointing
at their axioms = the understanding.
"""

import numpy as np

PHI = (1 + 5**0.5) / 2   # the golden ratio = the retention/retrieval gate


def phi_gate(magnitudes, phi=(1 + 5**0.5) / 2):
    """THE golden-ratio retention/retrieval gate — ONE definition, shared by the
    bundle and the retention decision. A lean/fiber is KEPT iff its holonomy
    magnitude >= 1/phi (~0.618). This is 'the gate that decides if something stays'
    (path of least resistance)."""
    return [m >= 1.0 / phi for m in magnitudes]


class HolonomyField:
    """Per-fiber accumulated directional memory. The retrieval component."""

    def __init__(self, net):
        self.net = net
        self.n_fibers = sum(W.shape[1] for W in net.frozen_W)
        self.H = np.zeros((self.n_fibers, self.n_fibers))  # H[f]=accumulated direction
        self.H_mag = np.zeros(self.n_fibers)               # accumulated magnitude
        # axioms = round-1 (first) fibers = the distance-0 core concepts
        self.axioms = list(range(net.frozen_W[0].shape[1]))

    def _dense(self, X):
        # forward_feats needs dense (N,D) one-hot; _tiny_word_data gives sparse (N,W)
        if X.ndim == 2 and X.shape[1] != self.net.D:
            from demo import wordlm
            return wordlm.one_hot(X, self.net.D // X.shape[1])
        return X

    def _locate(self, g):
        """Global fiber index -> (round, col). None if out of range."""
        col_g = 0
        for r, Wr in enumerate(self.net.frozen_W):
            n_r = Wr.shape[1]
            if g < col_g + n_r:
                return r, g - col_g
            col_g += n_r
        return None

    def _walk_to_axiom(self, f, max_hops=20):
        """LEARNED path-to-axiom: walk UP the stem from fiber f, at each step
        following the dominant source (argmax |W col|), until we reach a round-1
        fiber (an axiom, the distance-0 core concept). Returns the axiom's global
        index, or None if the path dead-ends at a raw input or loops (a lean that
        'does not work' — the phi-gate discards it because no target is written).
        Bounded (max_hops) and cycle-safe (revisit -> None)."""
        seen = set()
        cur = f
        for _ in range(max_hops):
            if cur in seen:
                return None  # cycle: a fiber that loops back
            seen.add(cur)
            loc = self._locate(cur)
            if loc is None:
                return None
            r, j = loc
            if r == 0:
                return cur  # round-1 fiber = an axiom
            Wr = self.net.frozen_W[r]
            col = np.abs(Wr[:, j])
            if col.sum() == 0:
                return None
            s = int(np.argmax(col))
            if s < self.net.D:
                return None  # dead-end at a raw input (no axiom reachable)
            cur = s - self.net.D
        return None

    def accumulate(self, X):
        """Grow the memory: LEANING IS THE LEARNING. Each firing fiber's lean is
        LEARNED from co-firing experience by walking UP the stem to the round-1
        AXIOM it compositionally reduces to (the 'short simplified path to the
        core concept'). A fiber that dead-ends at a raw input still accrues
        magnitude ('finding what does not work first') but writes NO target, so
        the phi-gate prunes it. Returns self."""
        from experiments.society import forward_feats
        Xd = self._dense(X)
        As = forward_feats(self.net, Xd)
        A = np.concatenate(As, 1) if As else np.zeros((len(Xd), 0))
        fired = (A > 0)
        col_g = 0
        for r, Wr in enumerate(self.net.frozen_W):
            n_r = Wr.shape[1]
            for j in range(n_r):
                col = np.abs(Wr[:, j])
                if col.sum() == 0:
                    continue
                rate = fired[:, col_g + j].mean()
                if rate <= 0:
                    continue
                self.H_mag[col_g + j] += rate
                # LEARN the lean: walk up the stem to the round-1 axiom.
                axiom = self._walk_to_axiom(col_g + j)
                if axiom is not None:
                    self.H[col_g + j, axiom] += rate
            col_g += n_r
        return self

    def lean_direction(self, fiber):
        """The lean of one fiber. Returns (unit_direction_vec, target_axiom, mag)
        or None if no accumulated direction. target_axiom = argmax of H row."""
        row = self.H[fiber]
        if row.sum() == 0:
            return None
        target = int(np.argmax(row))
        return row / (np.linalg.norm(row) + 1e-9), target, self.H_mag[fiber]

    def bundle(self, x):
        """RETRIEVAL: the understanding of input x = the bundle of every firing
        fiber's lean-direction vector, each pointing at its target axiom, gated
        by the PHI threshold (keep a lean iff its holonomy magnitude >= 1/PHI).
        Returns {vectors, axiom_targets, holonomy, n_leans}. Documented: this is
        a Python loop over fired fibers (vector-SHAPED output, not vectorized)."""
        from experiments.society import forward_feats
        Xd = self._dense(x.reshape(1, -1))
        As = forward_feats(self.net, Xd)
        A = np.concatenate(As, 1)[0] if As else np.zeros(self.n_fibers)
        fired = np.where(A > 0)[0]
        vecs, targets, mags = [], [], []
        for f in fired:
            d = self.lean_direction(f)
            if d is None:
                continue
            v, t, m = d
            if phi_gate([m])[0]:   # THE shared golden-ratio retention/retrieval gate (~0.618)
                vecs.append(v * m)
                targets.append(t)
                mags.append(m)
        return {"vectors": vecs, "axiom_targets": targets,
                "holonomy": mags, "n_leans": len(vecs)}

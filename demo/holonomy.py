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

    def accumulate(self, X):
        """Grow the memory: each fiber's firing on X adds a pull toward its
        dominant source (the fiber it reads most strongly). 'What happened
        because of the past.' Returns self."""
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
                s = int(np.argmax(col))
                target = -1 if s < self.net.D else s - self.net.D  # dominant source
                rate = fired[:, col_g + j].mean()
                if target >= 0 and rate > 0:
                    self.H[col_g + j, target] += rate
                self.H_mag[col_g + j] += rate
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
            if m >= 1.0 / PHI:   # THE golden-ratio retention/retrieval gate (~0.618)
                vecs.append(v * m)
                targets.append(t)
                mags.append(m)
        return {"vectors": vecs, "axiom_targets": targets,
                "holonomy": mags, "n_leans": len(vecs)}

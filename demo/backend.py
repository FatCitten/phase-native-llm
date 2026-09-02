"""demo/backend.py — vectorized tensor backend for Nautilus.

numpy is the default (CPU, zero deps). torch is optional: if importable, use it
so the same forward/backward code can run on a GPU. Swap by passing
`backend=TorchBackend()` to the net. All ops are the subset the hot loops need.
"""
from __future__ import annotations

import numpy as np


class NumpyBackend:
    name = "numpy"

    def matmul(self, a, b):
        return a @ b

    def gather(self, W, idx):
        """W (V, P), idx (N,) -> (N, P): the sparse embedding lookup."""
        return W[idx]

    def relu(self, x):
        return np.maximum(x, 0)

    def softmax(self, z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def argmax(self, x, axis=1):
        return x.argmax(axis)

    def mean(self, x):
        return float(x.mean())

    def sum(self, x, axis=None):
        return x.sum(axis)

    def concatenate(self, xs, axis=1):
        return np.concatenate(xs, axis)

    def zeros(self, shape):
        return np.zeros(shape)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def asarray(self, x):
        return np.asarray(x)

    def where(self, cond):
        return np.where(cond)[0]


class TorchBackend:
    """Optional GPU backend. Import-guarded: only constructed if torch is present."""
    name = "torch"

    def __init__(self, device="cuda"):
        import torch
        self.t = torch
        self.device = device if torch.cuda.is_available() else "cpu"

    def matmul(self, a, b):
        return self.t.matmul(self._t(a), self._t(b))

    def gather(self, W, idx):
        return self._t(W)[self._t(idx).long()]

    def relu(self, x):
        return self.t.relu(self._t(x))

    def softmax(self, z):
        z = self._t(z)
        e = self.t.exp(z - z.max(1, keepdims=True).values)
        return e / e.sum(1, keepdims=True)

    def argmax(self, x, axis=1):
        return self._t(x).argmax(axis)

    def mean(self, x):
        return float(self._t(x).mean())

    def sum(self, x, axis=None):
        return self._t(x).sum(axis)

    def concatenate(self, xs, axis=1):
        return self.t.cat([self._t(x) for x in xs], dim=axis)

    def zeros(self, shape):
        return self.t.zeros(shape)

    def zeros_like(self, x):
        return self.t.zeros_like(self._t(x))

    def asarray(self, x):
        return self._t(x)

    def where(self, cond):
        return self._t(cond).nonzero().squeeze(1).cpu().numpy()

    def _t(self, x):
        return x if isinstance(x, self.t.Tensor) else self.t.as_tensor(x, device=self.device)

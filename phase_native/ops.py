"""
Phase-native operations (Fourier Holographic Reduced Representation).

All ops are elementwise on unit-modulus complex vectors — O(dim), no gradients, no
matmul. This is the "no compute" substrate: binding is phase addition (the "connection"
from experiments/zkbundle_explicit_v2c.py, lifted from one circle to a vector of them);
readout is the Fourier/nearest-phase decode from that same file, generalised to vectors.
"""

from __future__ import annotations

import numpy as np


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Associate: elementwise complex multiply == per-channel phase addition."""
    return a * b


def unbind(m: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Query: multiply by the conjugate key == per-channel phase subtraction."""
    return m * np.conjugate(key)


def superpose(vectors) -> np.ndarray:
    """Store many bindings in ONE fixed-size vector (complex sum)."""
    return np.sum(np.stack(list(vectors)), axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised real part of the Hermitian inner product (phase alignment)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.real(np.vdot(a, b)) / denom)


def cleanup(vec: np.ndarray, codebook: np.ndarray) -> int:
    """Nearest-symbol readout by scanning a codebook: argmax_s Re<vec, c_s>.

    This is the O(V) baseline — provided so seek_scaling.py can contrast it with the
    O(1) CRT decode. The memory itself does NOT use this on the hot path.
    """
    scores = np.real(codebook @ np.conjugate(vec))  # (V,)
    return int(np.argmax(scores))

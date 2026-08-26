"""
Phase codebooks for the phase-native memory.

Two encodings, matching how the memory uses them:

* KEYS (cues) -> deterministic *random* phase vectors, generated on the fly from a
  stable hash of the cue string. Random phases are near-orthogonal, which minimises
  cross-talk between superposed bindings. Keys never need to be decoded, so there is
  no codebook to store and the key space is effectively unbounded.

* VALUES (conclusions) -> *CRT-structured* phase vectors over several coprime moduli.
  Each modulus m gets a block of `reps` channels, all carrying the phase
  2*pi*(v mod m)/m. A value integer is recovered channel-by-channel: circular-mean
  each modulus block, snap to the nearest residue, then Chinese-Remainder-combine the
  residues back into the integer. This decode costs O(#channels) and is INDEPENDENT of
  how many values exist (no scan of a vocabulary) -> the O(1)-in-vocab seek.

This generalises the single-circle Z_k phase trick proven in
`experiments/zkbundle_explicit_v2c.py` (that model is the D=1, single-modulus special
case) and reuses the pairwise CRT verified in `analysis/test_crt.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from math import gcd, prod

import numpy as np

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# Chinese Remainder Theorem (generalised to n coprime moduli)
# --------------------------------------------------------------------------------------
def crt_pair(a1: int, m1: int, a2: int, m2: int) -> int:
    """Solve x == a1 (mod m1), x == a2 (mod m2) for coprime m1, m2.

    Same construction as analysis/test_crt.py, kept here so the package is standalone.
    """
    inv_m1 = pow(m1, -1, m2)
    inv_m2 = pow(m2, -1, m1)
    M = m1 * m2
    return (a1 * m2 * inv_m2 + a2 * m1 * inv_m1) % M


def crt_combine(residues, moduli) -> int:
    """Combine residues under pairwise-coprime moduli into x in [0, prod(moduli))."""
    x, M = 0, 1
    for a, m in zip(residues, moduli):
        x = crt_pair(x, M, a % m, m)
        M *= m
    return x


def assert_coprime(moduli) -> None:
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(
                    f"moduli must be pairwise coprime; {moduli[i]} and {moduli[j]} "
                    f"share a factor. (CRT fails on shared factors — see minimal-handoff.md)"
                )


def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's builtin hash is per-process salted)."""
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


# --------------------------------------------------------------------------------------
# Keys: deterministic random phase vectors from a cue string
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def key_vector(cue: str, dim: int) -> np.ndarray:
    """Unit-modulus complex vector of length `dim`, deterministic in `cue`.

    Cached (the function is pure): the returned array is marked read-only so a stray write
    can't corrupt the cache. The ops (bind/unbind) allocate new arrays, so they never mutate
    it; make an explicit copy if you need a writable one.
    """
    rng = np.random.default_rng(stable_hash(cue) % (2**63))
    v = np.exp(1j * rng.uniform(0.0, TWO_PI, size=dim))
    v.flags.writeable = False
    return v


# --------------------------------------------------------------------------------------
# Values: CRT-structured phase vectors, decodable in O(#channels)
# --------------------------------------------------------------------------------------
@dataclass
class CRTValueCodebook:
    """Encode/decode integers in [0, capacity) as CRT phase vectors.

    dim = reps * len(moduli). Channel (i*reps + j) carries modulus moduli[i].
    """

    moduli: tuple[int, ...]
    reps: int = 32

    def __post_init__(self) -> None:
        self.moduli = tuple(int(m) for m in self.moduli)
        assert_coprime(self.moduli)
        self.capacity = prod(self.moduli)
        self.dim = self.reps * len(self.moduli)
        # per-channel modulus and the residue->phase lookup are precomputed for speed
        self._chan_mod = np.repeat(np.array(self.moduli), self.reps)  # (dim,)

    def encode(self, value: int) -> np.ndarray:
        """Integer -> unit complex vector of length dim."""
        if not (0 <= value < self.capacity):
            raise ValueError(f"value {value} out of range [0,{self.capacity})")
        residues = np.array([value % m for m in self.moduli])          # (n_mod,)
        phases = TWO_PI * np.repeat(residues / np.array(self.moduli), self.reps)
        return np.exp(1j * phases)

    def decode(self, vec: np.ndarray) -> int:
        """Noisy phase vector -> nearest integer, O(dim), no vocabulary scan."""
        residues = []
        for i, m in enumerate(self.moduli):
            block = vec[i * self.reps : (i + 1) * self.reps]
            # circular mean of the block's phases (noise-averaging over reps)
            mean_angle = np.angle(np.sum(block)) % TWO_PI
            r = int(np.rint(mean_angle / TWO_PI * m)) % m
            residues.append(r)
        return crt_combine(residues, self.moduli)

    def nbytes_per_value(self) -> int:
        """A value's low-byte footprint = its residue tuple (1 byte per small modulus)."""
        return len(self.moduli)


DEFAULT_MODULI = (8, 9, 5, 7, 11, 13)  # pairwise coprime; capacity = 360360

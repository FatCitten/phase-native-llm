"""
PhaseNuggetMemory — the LLM's O(1) "self-agent" memory.

A single fixed-size complex vector M holds every association by superposition:

    M = sum_j  bind(key(cue_j), value(conclusion_j))

* write(cue, conclusion): solidify an established result into M (LLM-guided).
* recall(cue): unbind M by the cue's key, CRT-decode the value -> O(dim), constant in
  the number of stored nuggets N and in the vocabulary V. Returns a confidence so the
  caller (the LLM) can VERIFY and re-derive if interference corrupted the recall.
* forget(cue): subtract exactly that binding back out of M (algebraic edit).

M is fixed size regardless of N. The only per-nugget state is bookkeeping: a cue->id map
(small ints) used for provenance and exact forgetting, and an id->payload table that
renders a decoded value id back to text. Neither is consulted on the recall hot path.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

import numpy as np

from .codebook import DEFAULT_MODULI, CRTValueCodebook, key_vector
from .ops import bind, unbind


@dataclass
class RecallResult:
    hit: bool
    payload: object | None
    value_id: int
    confidence: float  # normalized matched-filter score; ~1.0 = clean hit, ~0.3 = noise

    def __bool__(self) -> bool:
        return self.hit


@dataclass
class PhaseNuggetMemory:
    moduli: tuple[int, ...] = (8, 9, 5, 7)
    reps: int = 512
    min_confidence: float = 0.45

    def __post_init__(self) -> None:
        self.values = CRTValueCodebook(self.moduli, self.reps)
        self.dim = self.values.dim
        self.M = np.zeros(self.dim, dtype=np.complex128)
        self._payload2id: dict[str, int] = {}
        self._id2payload: dict[int, object] = {}
        self._cue2id: dict[str, int] = {}  # provenance + exact-forget bookkeeping only
        self._cb: np.ndarray | None = None  # cached cleanup codebook (built lazily)

    # -- payload id assignment (sequential -> no collisions, well-spread residues) -----
    def _value_id(self, payload) -> int:
        skey = json.dumps(payload, sort_keys=True, default=str)
        if skey not in self._payload2id:
            new_id = len(self._payload2id)
            if new_id >= self.values.capacity:
                raise ValueError(
                    f"value vocabulary exceeded CRT capacity {self.values.capacity}; "
                    f"add moduli"
                )
            self._payload2id[skey] = new_id
            self._id2payload[new_id] = payload
            self._cb = None  # invalidate cleanup codebook cache
        return self._payload2id[skey]

    def _cleanup_codebook(self) -> np.ndarray:
        """(V, dim) matrix of value vectors, cached until a new value appears."""
        if self._cb is None:
            V = len(self._id2payload)
            self._cb = (
                np.stack([self.values.encode(i) for i in range(V)])
                if V
                else np.zeros((0, self.dim), dtype=np.complex128)
            )
        return self._cb

    def _binding(self, cue: str, value_id: int) -> np.ndarray:
        return bind(key_vector(cue, self.dim), self.values.encode(value_id))

    # -- core API ----------------------------------------------------------------------
    def write(self, cue: str, conclusion) -> int:
        """Solidify (cue -> conclusion) into M. Rewriting a cue replaces its binding."""
        if cue in self._cue2id:  # subtract the stale binding first
            self.M -= self._binding(cue, self._cue2id[cue])
        vid = self._value_id(conclusion)
        self.M += self._binding(cue, vid)
        self._cue2id[cue] = vid
        return vid

    def recall(self, cue: str, mode: str = "cleanup") -> RecallResult:
        """Associative seek by a string cue (hashed to a decorrelated key). O(dim) in N.

        mode="cleanup" (default): matched-filter decode over the value alphabet — O(V) in the
            answer vocabulary, still O(1) in N; its normalized score is a real confidence.
        mode="crt": per-channel CRT residue decode — O(dim), constant in N and V. Lower capacity.
        Confidence = normalized score Re<value(vid), est>/dim (~1.0 clean hit, ~0.3 noise).
        """
        return self.recall_key(key_vector(cue, self.dim), mode)

    def recall_key(self, key: np.ndarray, mode: str = "cleanup") -> RecallResult:
        """Recall from a key VECTOR directly (fuzzy/semantic cues supply their own key)."""
        est = unbind(self.M, key)
        if mode == "crt":
            vid = self.values.decode(est)
        elif mode == "cleanup":
            cb = self._cleanup_codebook()
            if cb.shape[0] == 0:
                return RecallResult(False, None, -1, 0.0)
            vid = int(np.argmax(np.real(cb @ np.conjugate(est))))
        else:
            raise ValueError(f"unknown recall mode {mode!r}")
        confidence = float(np.real(np.vdot(self.values.encode(vid), est)) / self.dim)
        payload = self._id2payload.get(vid)
        hit = payload is not None and confidence >= self.min_confidence
        return RecallResult(hit=hit, payload=payload, value_id=vid, confidence=confidence)

    def write_key(self, key: np.ndarray, conclusion) -> int:
        """Solidify (key vector -> conclusion) into M. Used by fuzzy/semantic cue layers."""
        vid = self._value_id(conclusion)
        self.M += bind(key, self.values.encode(vid))
        return vid

    def forget(self, cue: str) -> bool:
        """Algebraic edit: subtract exactly this cue's binding out of M."""
        if cue not in self._cue2id:
            return False
        self.M -= self._binding(cue, self._cue2id[cue])
        del self._cue2id[cue]
        return True

    # -- introspection / control -------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_nuggets": len(self._cue2id),
            "n_distinct_values": len(self._id2payload),
            "dim": self.dim,
            "moduli": list(self.moduli),
            "capacity_values": self.values.capacity,
            "M_bytes_fixed": self.M.nbytes,
            "logical_nugget_bytes": self.values.nbytes_per_value(),
        }

    # -- low-byte serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "moduli": list(self.moduli),
            "reps": self.reps,
            "min_confidence": self.min_confidence,
            "M_b64": base64.b64encode(self.M.astype(np.complex128).tobytes()).decode(),
            "id2payload": {str(k): v for k, v in self._id2payload.items()},
            "cue2id": self._cue2id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhaseNuggetMemory":
        mem = cls(tuple(d["moduli"]), int(d["reps"]), float(d["min_confidence"]))
        mem.M = np.frombuffer(base64.b64decode(d["M_b64"]), dtype=np.complex128).copy()
        mem._id2payload = {int(k): v for k, v in d["id2payload"].items()}
        mem._payload2id = {
            json.dumps(v, sort_keys=True, default=str): int(k)
            for k, v in d["id2payload"].items()
        }
        mem._cue2id = dict(d["cue2id"])
        return mem

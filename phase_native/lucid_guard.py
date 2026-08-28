"""
LucidMemory — verify-before-assert for an agent's own prior conclusions.

The failure this targets fires when the agent does NOT know to look: a conclusion it derived
earlier resurfaces, and recall feels identical to improvisation. So the check must be cheap
enough to run on *every* about-to-be-asserted prior claim:

    commit(statement)  -> solidify a conclusion the moment you make it (auto-checkpoint)
    verify(claim)      -> "recalled (here is the receipt)"  OR  "unknown — you're improvising"

`verify` returns a receipt (the decoded value's CRT residues) you can re-derive and check —
an audit trail a cosine score cannot give — or it abstains. Fuzzy cue matching (semantic.py)
means an approximate restatement still recalls the original.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory import PhaseNuggetMemory
from .semantic import text_key


@dataclass
class VerifyResult:
    status: str  # "recalled" | "unknown"
    statement: object | None  # the stored conclusion, if recalled
    confidence: float
    receipt: dict | None  # {value_id, residues} — re-derivable audit trail

    def __bool__(self) -> bool:
        return self.status == "recalled"


class LucidMemory:
    def __init__(self, reps: int = 512, moduli=(8, 9, 5, 7), min_margin: float = 0.12):
        # min_margin abstains when two stored conclusions match almost equally (a genuine tie).
        # It does NOT resolve semantic opposites that share vocabulary ("nullable after" vs
        # "non-nullable before") — those are inseparable with lexical keys and need a semantic
        # embedding front-end. See experiments/beat_the_file.py (near-collision row).
        self.mem = PhaseNuggetMemory(moduli=moduli, reps=reps, min_margin=min_margin)
        self.dim = self.mem.dim

    def commit(self, statement: str) -> int:
        """Checkpoint a conclusion at the moment you commit to it."""
        return self.mem.write_key(text_key(statement, self.dim), statement)

    def verify(self, claim: str) -> VerifyResult:
        """Before asserting a remembered claim: is it recalled (with a receipt) or improvised?"""
        r = self.mem.recall_key(text_key(claim, self.dim))
        if not r.hit:
            return VerifyResult("unknown", None, r.confidence, None)
        residues = [r.value_id % m for m in self.mem.moduli]
        return VerifyResult("recalled", r.payload, r.confidence,
                            {"value_id": r.value_id, "residues": residues})

    def guarded(self, claim: str) -> str:
        """One-line verdict an agent can gate an assertion on."""
        v = self.verify(claim)
        if v:
            return f"RECALLED (conf {v.confidence:.2f}, receipt {v.receipt['residues']}): {v.statement}"
        return "UNKNOWN — not in memory; do not assert as recalled, re-derive."

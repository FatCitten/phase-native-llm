"""
Fuzzy cue encoding — recall a prior conclusion from an *approximate* restatement.

An exact string cue (hashed) only fires on a byte-identical cue — no better than grep. To beat
grep we need cues where "close in meaning" maps to "close key." We do it without an ML embedder:
a cue's key is the superposition of its content words' phase vectors, normalized per channel.

- Two restatements sharing most content words -> highly overlapping token sets -> nearly the
  same key -> recalled via the memory's phase-noise robustness.
- Two *distinct* conclusions -> mostly disjoint tokens -> decorrelated keys -> no crosstalk,
  and an unrelated query abstains.

Honest scope: this matches on *lexical* overlap (word reordering, filler, casing, morphology-lite),
which is exactly where substring grep fails. True synonymy ("retry" ~ "back off") shares no tokens
and needs an embedding front-end — a drop-in replacement for `text_key` (same interface).
"""

from __future__ import annotations

import re

import numpy as np

from .codebook import key_vector

_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "we", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "for", "in", "on", "at", "by", "with", "as",
    "because", "so", "then", "than", "over", "under", "into", "our", "us", "i", "you", "he",
    "she", "they", "them", "not", "but", "if", "chose", "chosen", "decided", "went", "use",
    "used", "using", "instead", "since", "due", "will", "would", "can", "could", "do", "did",
}


def content_tokens(text: str) -> list[str]:
    """Lowercased, de-duplicated content words (stopwords and pure numbers dropped)."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    seen, out = set(), []
    for t in toks:
        if t in _STOP or t.isdigit() or len(t) <= 1:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def text_key(text: str, dim: int) -> np.ndarray:
    """Unit-modulus phase key = per-channel-normalized superposition of the token phasors."""
    toks = content_tokens(text)
    if not toks:
        return key_vector(text, dim)  # fall back to exact-string key
    k = np.zeros(dim, dtype=np.complex128)
    for t in toks:
        k = k + key_vector(t, dim)
    mag = np.abs(k)
    mag[mag == 0.0] = 1.0
    return k / mag


def jaccard(a: str, b: str) -> float:
    """Content-token Jaccard overlap — used only for baselines/analysis, not by the memory."""
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

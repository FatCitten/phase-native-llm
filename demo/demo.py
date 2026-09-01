"""demo/demo.py — the Nautilus machine: shared helpers for the Extension VI demo package.

NAUTILUS: a consolidation network whose structure is legible (observe/trace), editable
(edit), and durable (save/load/read/write). This module provides the shared corpus loader
used by the char-LM, the StructureEngine, and the LLM-playwright harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_sections(path):
    """Read a corpus file and split it into named sections by '=== NAME ===' markers."""
    sections = {}
    current = None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("===") and s.endswith("==="):
            current = s.strip("= ").strip()
            sections[current] = []
        elif s.startswith("#") or not s:
            continue
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}

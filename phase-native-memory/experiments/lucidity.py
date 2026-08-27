"""
Lucidity: a memory that never lies to the model, and knows when it doesn't know.

The AI-native significance metric. Every similarity/RAG memory returns a nearest match for ANY
query — it cannot certify "I actually stored this, exactly." This memory can. We measure:

  true recall    = known fact returned correctly AND confidence >= gate  (it genuinely knows)
  confabulation  = confidently WRONG on a known cue, OR confidently returns a hit for an
                   UNKNOWN cue it never stored  (it lied)
  abstention     = confidence < gate  (it honestly says "I don't know — re-derive")

Result: within a tunable, SELF-DIAGNOSED capacity, confabulation is 0% — and when the memory
is pushed past capacity it converts uncertainty into honest abstention (and the known-vs-unknown
confidence margin crosses zero, so the model can SEE it is leaving the lucid zone).

Run: python experiments/lucidity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_native.memory import PhaseNuggetMemory

RESULTS = Path("results")


def profile(reps, N, n_unknown=300):
    """Return recall/confab/abstain rates and the known-vs-unknown confidence margin."""
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=reps)  # gate = min_confidence = 0.45
    for i in range(N):
        mem.write(f"fact_{i}", i)
    correct = confab = abstain = 0
    known_conf = []
    for i in range(N):
        r = mem.recall(f"fact_{i}")
        if not r.hit:
            abstain += 1
        elif r.payload == i:
            correct += 1
            known_conf.append(r.confidence)
        else:
            confab += 1
    unk_hit = 0
    unk_conf = []
    for j in range(n_unknown):
        r = mem.recall(f"unknown_{j}")
        unk_conf.append(r.confidence)
        if r.hit:  # claims to know something it never stored
            unk_hit += 1
    tot = N + n_unknown
    return {
        "dim": mem.dim,
        "true_recall": correct / N,
        "confab": (confab + unk_hit) / tot,
        "abstain_known": abstain / N,
        "margin": (min(known_conf) if known_conf else 0.0) - max(unk_conf),
    }


def lucid_capacity(reps, grid=range(5, 260, 5)):
    """Largest load with 100% true recall AND 0% confabulation."""
    cap, dim = 0, None
    for N in grid:
        p = profile(reps, N)
        dim = p["dim"]
        if p["true_recall"] == 1.0 and p["confab"] == 0.0:
            cap = N
        elif cap:  # already found the edge
            break
    return dim, cap


def main():
    RESULTS.mkdir(exist_ok=True)

    # Panel A: lucid (zero-confabulation) capacity scales linearly with memory size
    caps = [lucid_capacity(reps) for reps in (256, 512, 1024, 2048)]
    kb = [d * 16 / 1024 for d, _ in caps]
    facts = [c for _, c in caps]
    bytes_per_fact = [(d * 16) / c if c else float("nan") for d, c in caps]
    print("=== A: zero-confabulation 'lucid capacity' vs memory size ===")
    for (d, c), b in zip(caps, bytes_per_fact):
        print(f"  {d*16/1024:5.0f} KB (dim {d:5d})  ->  {c:4d} verifiable facts   ({b:.0f} bytes/fact)")

    # Panel B: at a fixed size, confabulation stays 0 in the lucid zone; margin self-diagnoses
    reps_b = 512  # dim 2048 = 32 KB
    loads = list(range(10, 205, 10))
    rows = [profile(reps_b, N) for N in loads]
    dimB = rows[0]["dim"]
    print(f"\n=== B: dim {dimB} ({dimB*16/1024:.0f} KB) — confabulation vs load ===")
    for N, r in zip(loads, rows):
        print(f"  load={N:4d}  recall={r['true_recall']*100:5.1f}%  confab={r['confab']*100:4.1f}%  "
              f"abstain={r['abstain_known']*100:4.1f}%  margin={r['margin']:+.2f}")

    json.dump({"capacity": [{"dim": d, "kb": d * 16 / 1024, "facts": c} for d, c in caps],
               "bytes_per_fact": bytes_per_fact,
               "sweep_dim": dimB, "loads": loads, "rows": rows},
              open(RESULTS / "lucidity.json", "w"), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(kb, facts, "o-", color="#2e7d32")
    for x, y in zip(kb, facts):
        ax1.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.set_xlabel("memory size (KB, fixed)")
    ax1.set_ylabel("verifiable facts held (0% confabulation)")
    ax1.set_title("Lucid capacity scales linearly (~640 bytes / verifiable fact)")
    ax1.grid(True, alpha=0.3)

    conf = [r["confab"] * 100 for r in rows]
    marg = [r["margin"] for r in rows]
    lucid = [N for N, r in zip(loads, rows) if r["true_recall"] == 1.0 and r["confab"] == 0.0]
    if lucid:
        ax2.axvspan(loads[0], max(lucid), color="#2e7d32", alpha=0.10, label="lucid zone (0% confab)")
    ax2.plot(loads, conf, "o-", color="#c62828", label="confabulation %")
    ax2.set_xlabel(f"facts stored (in {dimB*16/1024:.0f} KB)")
    ax2.set_ylabel("confabulation %", color="#c62828")
    ax2.set_title("0% confabulation in the lucid zone; the margin flags the edge")
    axr = ax2.twinx()
    axr.plot(loads, marg, "s--", color="#1565c0", label="confidence margin")
    axr.axhline(0, color="#1565c0", ls=":", alpha=0.5)
    axr.set_ylabel("known−unknown confidence margin", color="#1565c0")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "lucidity.png", dpi=140)
    print("\nSaved results/lucidity.{json,png}")


if __name__ == "__main__":
    main()

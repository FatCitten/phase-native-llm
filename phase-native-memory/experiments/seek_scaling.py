"""
O(1) seek benchmark + capacity/fidelity — the core "seek does not slow down" claim.

Two results, saved to results/:

1. SEEK TIME vs N (number of stored nuggets), log-log:
     * phase memory (unbind + CRT decode) -> FLAT in N   [O(1)]
     * naive explicit store + similarity scan -> LINEAR in N   [O(N)]
   Superposing bindings into one fixed vector is what buys constant-time recall.

2. RECALL ACCURACY vs load, for the two decode modes:
     * CRT decode  (O(1) in the value vocabulary too) -> lower capacity
     * cleanup     (O(V), matched filter)             -> higher capacity
   Capacity is bounded and tunable via dim; this is measured, not hidden.

Pure CPU, no API. Run: python experiments/seek_scaling.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_native.codebook import key_vector
from phase_native.memory import PhaseNuggetMemory
from phase_native.ops import unbind

RESULTS = Path("results")


def time_phase_seek(dim_reps=1024, loads=(10, 30, 100, 300, 1000, 3000), trials=300):
    """Median wall-time of one phase recall (unbind+CRT decode) as N grows."""
    out = []
    for N in loads:
        mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=dim_reps)
        for i in range(N):
            mem.write(f"cue_{i}", i % mem.values.capacity)
        cues = [f"cue_{i}" for i in np.random.default_rng(0).integers(0, N, trials)]
        t0 = time.perf_counter()
        for c in cues:
            mem.recall(c, mode="crt")
        dt = (time.perf_counter() - t0) / trials
        out.append((N, dt))
    return out


def time_naive_seek(dim, loads=(10, 30, 100, 300, 1000, 3000), trials=300):
    """Median wall-time of a naive explicit-store similarity scan (O(N))."""
    out = []
    rng = np.random.default_rng(1)
    for N in loads:
        keys = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(N, dim)))
        values = np.arange(N)
        idx = rng.integers(0, N, trials)
        t0 = time.perf_counter()
        for q in idx:
            scores = np.real(keys @ np.conjugate(keys[q]))  # scan all N stored keys
            _ = values[int(np.argmax(scores))]
        dt = (time.perf_counter() - t0) / trials
        out.append((N, dt))
    return out


def capacity_curve(reps=512, loads=(10, 25, 50, 100, 200, 400), trials_seed=0):
    """Recall accuracy vs load for CRT (O(1)) and cleanup (O(V)) decode."""
    rows = []
    for N in loads:
        mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=reps)
        facts = {f"cue_{i}": i for i in range(N)}
        for c, p in facts.items():
            mem.write(c, p)
        crt = sum(mem.recall(c, mode="crt").payload == p for c, p in facts.items()) / N
        clean = sum(mem.recall(c, mode="cleanup").payload == p for c, p in facts.items()) / N
        rows.append({"N": N, "crt": crt, "cleanup": clean, "dim": mem.dim})
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    dim = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=1024).dim
    print(f"dim = {dim}")

    phase = time_phase_seek(dim_reps=1024)
    naive = time_naive_seek(dim=dim)
    cap = capacity_curve(reps=512)

    print("\nSEEK TIME (microseconds/recall):")
    print(f"{'N':>6} {'phase O(1)':>12} {'naive O(N)':>12}")
    for (N, tp), (_, tn) in zip(phase, naive):
        print(f"{N:>6} {tp*1e6:>12.1f} {tn*1e6:>12.1f}")

    print("\nCAPACITY (recall accuracy):")
    print(f"{'N':>6} {'CRT O(1)':>10} {'cleanup O(V)':>13}")
    for r in cap:
        print(f"{r['N']:>6} {r['crt']:>10.2f} {r['cleanup']:>13.2f}")

    results = {
        "dim": dim,
        "seek_phase_us": [[N, t * 1e6] for N, t in phase],
        "seek_naive_us": [[N, t * 1e6] for N, t in naive],
        "capacity": cap,
    }
    (RESULTS / "seek_scaling.json").write_text(json.dumps(results, indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    Np = [N for N, _ in phase]
    ax1.loglog(Np, [t * 1e6 for _, t in phase], "o-", label="phase memory  O(1)")
    ax1.loglog(Np, [t * 1e6 for _, t in naive], "s--", label="naive scan  O(N)")
    ax1.set_xlabel("nuggets stored (N)")
    ax1.set_ylabel("seek time (microseconds)")
    ax1.set_title("Seek time does not grow with what you store")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    Nc = [r["N"] for r in cap]
    ax2.plot(Nc, [r["crt"] for r in cap], "o-", label="CRT decode  O(1) in vocab")
    ax2.plot(Nc, [r["cleanup"] for r in cap], "s-", label="cleanup  O(V), higher capacity")
    ax2.axhline(0.9, color="gray", ls=":", alpha=0.6)
    ax2.set_xlabel("nuggets stored (N)")
    ax2.set_ylabel("recall accuracy")
    ax2.set_title(f"Bounded capacity, tunable via dim (dim={cap[0]['dim']})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "seek_scaling.png", dpi=140)
    print(f"\nSaved results/seek_scaling.json and results/seek_scaling.png")


if __name__ == "__main__":
    main()

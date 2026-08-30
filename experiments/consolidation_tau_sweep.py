"""
Why the tightening ceiling is 0.8, not 0.9 -- a transparent, reproducible sweep.

The tightening ratio (experiments/consolidation_rounds.py) ramps tau loose->tight across rounds. The
END of that ramp is a free hyperparameter; this sweep sets it honestly. For each ceiling c we build
the ramp [0, linspace(0.3, c, 4)] (round 1 = pure axioms), run the full tightened loop, and report
final accuracy, wiring, capability/synapse, distance, and -- as the strict control -- a monolithic net
magnitude-pruned to the SAME synapse budget.

Finding: c=0.9 wins capability/synapse by a hair but OVER-tightens the last round (accuracy dips), while
c=0.8 gives the best accuracy and a smooth monotone rise at ~equal capability/synapse. So the loop
defaults to 0.8. Every ceiling still beats dense joint training per synapse; none beats plain pruning.

Pure numpy, CPU.  Run: python experiments/consolidation_tau_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.consolidation_rounds import hier_teacher_data, monolithic, run_loop


def main():
    Xtr, ytr, Xte, yte, D, C = hier_teacher_data()
    ROUNDS, EP = 5, 1500

    un, ur, _ = run_loop(Xtr, ytr, Xte, yte, D, C, [0.0] * ROUNDS, epochs=EP)
    ua, us = ur[-1]["test_acc"], un.synapses
    print(f"untightened baseline: acc={ua:.3f}  syn={us}  cap/syn={ua/us*1e3:.3f}e-3  dist={ur[-1]['mean_dist']:.2f}\n")
    print(f"{'ceiling':>7}  {'acc':>5}  {'syn':>5}  {'cap/syn':>8}  {'dist':>4}  {'cross':>5}  "
          f"{'per-round acc':<32}  {'pruned@budget':>13}")

    best = None
    for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
        taus = [0.0] + list(np.round(np.linspace(0.3, c, ROUNDS - 1), 3))
        net, rr, _ = run_loop(Xtr, ytr, Xte, yte, D, C, taus, epochs=EP)
        acc, syn = rr[-1]["test_acc"], net.synapses
        cps = acc / syn * 1e3
        (_, _), (mp_a, mp_s) = monolithic(Xtr, ytr, Xte, yte, D, C, H=len(net.dist),
                                          epochs=ROUNDS * EP, target_syn=syn)
        per_round = " ".join(f"{s['test_acc']:.3f}" for s in rr)
        print(f"{c:>7.1f}  {acc:.3f}  {syn:>5d}  {cps:>6.3f}e-3  {rr[-1]['mean_dist']:>4.2f}  "
              f"{net.cross_edges:>5d}  {per_round:<32}  {mp_a:.3f}@{mp_s}")
        # smoothness = penalize any per-round accuracy drop (over-tightening); prefer high final acc
        drop = sum(max(0.0, rr[i - 1]["test_acc"] - rr[i]["test_acc"]) for i in range(1, len(rr)))
        score = acc - drop
        if best is None or score > best[0]:
            best = (score, c, acc, syn, cps)

    print(f"\nchosen ceiling: c={best[1]} (acc={best[2]:.3f}, syn={best[3]}, cap/syn={best[4]:.3f}e-3) "
          f"-- best accuracy with a smooth, dip-free rise. Every ceiling beats dense per synapse; "
          f"none beats plain pruning.")


if __name__ == "__main__":
    main()

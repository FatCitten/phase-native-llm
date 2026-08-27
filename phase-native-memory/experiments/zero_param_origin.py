"""
The origin story, in ~40 lines of numpy: 100% generalization with ZERO parameters.

This is where the whole idea starts. Modular addition `(a + b) mod k` is the task that
"grokking" papers train networks for thousands of steps to suddenly solve. But the task IS
angle addition on a circle, so if you *encode* it that way there is nothing to learn:

    a, b        -> phases  2*pi*a/k, 2*pi*b/k          (put the integers on a circle)
    connection  -> result_phase = phase_a + phase_b    (the group operation = binding)
    readout     -> logits[c] = cos(result_phase - 2*pi*c/k)   (Fourier / nearest-phase decode)
    answer      -> argmax_c logits[c] = (a + b) mod k

100% train AND test accuracy at step 0, with 0 learnable parameters and 0 gradient updates.
No torch, no optimizer, no training loop — the geometry is the solution.

This is exactly the **D=1, single-binding special case** of `phase_native.PhaseNuggetMemory`:
one circle instead of a vector of them; `bind` (phase add) is the connection here; the Fourier
decode here is the memory's `cleanup`/CRT readout. The memory lifts this one trick to a vector
of circles so an LLM can superpose *many* associations into one fixed-size store and recall any
of them in O(1). Run it and see the zero-parameter table for yourself.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def solve_mod_add(k: int) -> tuple[float, int]:
    """Return (accuracy over all k*k pairs, learnable_params) for (a+b) mod k. No training."""
    a = np.repeat(np.arange(k), k)
    b = np.tile(np.arange(k), k)
    truth = (a + b) % k

    # encode integers as phases, add the phases (the "connection"), decode by Fourier basis
    result_phase = TWO_PI * a / k + TWO_PI * b / k                 # (k*k,)
    class_phase = TWO_PI * np.arange(k) / k                        # (k,)
    logits = np.cos(result_phase[:, None] - class_phase[None, :])  # (k*k, k)
    pred = logits.argmax(axis=1)

    accuracy = float((pred == truth).mean())
    learnable_params = 0
    return accuracy, learnable_params


def main() -> None:
    print("=" * 60)
    print("ZERO-PARAMETER GROKKING  —  100% at step 0, no training")
    print("=" * 60)
    print(f"\n{'k':>4} {'train+test acc':>16} {'learnable params':>18} {'grad steps':>12}")
    all_perfect = True
    for k in (11, 17, 23, 29, 31, 37):
        acc, params = solve_mod_add(k)
        all_perfect &= acc == 1.0
        print(f"{k:>4} {acc*100:>15.2f}% {params:>18} {0:>12}")

    print("\nFor comparison, grokking papers report the SAME task solved only after:")
    print("  - Power et al. (2021): ~10,000-20,000 gradient steps")
    print("  - Gromov (2023):       ~5,000-15,000 gradient steps")
    print("\nThe difference is not a better optimizer — it is the right primitive.")
    print("Grokking is the COST of discovering geometric structure from flat embeddings;")
    print("hand the model the geometry and the cost is zero.")
    print("\nphase_native.PhaseNuggetMemory is this exact trick lifted from one circle to a")
    print("vector of circles, so an LLM can superpose many associations and recall in O(1).")
    print("\n" + ("ALL k PERFECT (100%, 0 params, 0 steps)." if all_perfect else "unexpected: not all perfect"))


if __name__ == "__main__":
    main()

# PHASE TRANSITION SCAN RESULTS
**Date:** March 29, 2026
**Experiment:** Multi-seed parity training for n = [32, 40, 48, 52, 56, 58, 60, 64, 72, 80]
**Seeds:** 5 per n value
**Samples:** 2000 per run
**Epochs:** 400

---

## KEY FINDING: Soft Phase Transition

The transition from ordered (learnable) to disordered (not learnable) is NOT a hard boundary but a **continuous phase transition** with a critical region.

| n   | Mean Acc | Success Rate | Phase Std | Phase     |
|-----|----------|--------------|-----------|-----------|
| 32  | 100%     | 5/5          | 0.00      | Ordered   |
| 40  | 100%     | 5/5          | 0.00      | Ordered   |
| 48  | 100%     | 5/5          | 0.00      | Ordered   |
| 52  | 89.7%    | 4/5          | 0.27      | Critical  |
| 56  | 69.3%    | 2/5          | 0.83      | Critical  |
| 58  | 70.3%    | 2/5          | 0.87      | Critical  |
| 60  | 59.0%    | 1/5          | 1.17      | Critical  |
| 64  | 60.4%    | 1/5          | 1.12      | Critical  |
| 72  | 49.7%    | 0/5          | 1.41      | Disordered|
| 80  | 51.3%    | 0/5          | 1.39      | Disordered|

---

## THREE REGIMES IDENTIFIED

1. **STABLE PHASE (n <= 48):**
   - 100% success rate across all seeds
   - Phase std = 0 (perfect convergence to pi)
   - Network reliably learns parity

2. **CRITICAL REGION (n = 52-64):**
   - Success becomes probabilistic (40-90%)
   - Phase std increases from 0.3 to 1.2
   - Sensitive to initialization
   - This is the "phase transition zone"

3. **DISORDERED PHASE (n >= 72):**
   - ~50% accuracy (random guessing)
   - Phase std ~1.4 (fully disordered)
   - Cannot learn regardless of initialization

---

## PHYSICS INTERPRETATION

This is a genuine phase transition phenomenon:
- Order parameter: phase_std (measures coherence of bit phases)
- Critical region: where success probability drops
- n* ~ 60-65: approximate transition point

The smooth increase in phase_std from 0 to 1.4 is characteristic of a continuous phase transition, similar to:
- Magnetization near Curie temperature
- Water density at boiling point
- Crystal formation from supercooled liquid

---

## NEXT STEPS

1. Finer scan around n = 52-64 to find exact n*
2. Test longer training (more epochs) in critical region
3. Try different learning rates to see if they affect n*
4. Measure critical exponent beta more precisely

---

## FILES

- refined_transition.py: Main experiment script
- refined_transition.png: Visualization
- refined_transition_results.json: Raw data

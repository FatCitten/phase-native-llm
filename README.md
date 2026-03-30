# Phase-Native LLM

Geometric phase encoding for neural networks.

## Quick Start

See `handoff.md` for detailed experiment guide.

## Control Experiment (v3) — Random LUT

The **control_v3** experiments test whether ZkBundle architecture requires GROUP STRUCTURE or can learn arbitrary deterministic patterns (random LUT).

### Key Files
| File | Description |
|------|-------------|
| `experiments/control_v3_step1_mul_mod_k.py` | Step 1: Fixed variance |
| `experiments/control_v3_step2_ceiling_test.py` | Step 2: Ceiling test |
| `experiments/control_v3_step3_heatmap.py` | Step 3: Heatmap diagnostic |
| `experiments/control_v3_step4_phase_resolution_scaling.py` | Step 4: Scaling analysis |
| `experiments/control_v3_step5_k13_k17_scaling.py` | Step 5: k=13,17 extension |
| `experiments/control_v3_step6_normalization_fix.py` | Step 6: Fix normalization bug |
| `experiments/control_v3_step7_k13_k17_retrain.py` | Step 7: Retrain with consistent normalization |

### Results
| k | theta | beta | R² |
|---|-------|------|-----|
| 5 | 0.069 | 0.168 | 0.999 |
| 7 | 0.134 | 0.155 | 0.991 |
| 11 | 0.141 | 0.148 | 0.902 |
| 13 | 0.318 | 0.168 | 0.974 |
| 17 | 0.452 | 0.264 | 0.960 |

- **Scaling**: Power law (k^1.72, R²=0.92) beats constant model (R²=0.00)
- **Beta**: NOT universal (std=0.042 > 0.015) — inconsistent across k
- **Normalization**: Fixed inconsistent x-axis bug (k=5,7,11 vs k=13,17)

### Key Finding: DIAGONAL FAILURE
Model learns "a ≠ b → pick larger" perfectly (100% on upper/lower triangles) but fails on diagonal (a == b) because phase addition cannot resolve when inputs are equal.

## Verified Experiments

| File | Description |
|------|-------------|
| `experiments/valid/phase_convergence.py` | Phase uniformity check |
| `experiments/valid/z_4_generalization.py` | Z_4 single network learning |
| `experiments/valid/scaling_law.py` | σ* × k scaling law |
| `experiments/valid/z_6_composite.py` | Z_6 composite test |

## Key Findings

- **σ* × k ≈ 1.79** (asymptote from fit)
- **Z_6 works** - supports "not prime-power" rule
- **CRT** requires coprime factors

## Structure

```
experiments/
  control_v3/     - Random LUT control experiments
  valid/          - Reproducible results
  invalid/        - Known issues
  legacy/         - Historical
results/control_v3/ - Control experiment outputs
analysis/        - Debug/measure
scratch/         - Exploratory
```

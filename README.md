# Phase-Native LLM

Geometric phase encoding for neural networks.

## Quick Start

See `handoff.md` for detailed experiment guide.

## Experiments

### Verified (Use These)
| File | Description |
|------|-------------|
| `experiments/valid/phase_convergence.py` | Phase uniformity check |
| `experiments/valid/z_4_generalization.py` | Z_4 single network learning |
| `experiments/valid/scaling_law.py` | σ* × k scaling law |
| `experiments/valid/z_6_composite.py` | Z_6 composite test |

### Invalid
| File | Issue |
|------|-------|
| `experiments/invalid/nongroup_control_INVALID.py` | Mod is vacuous - not a real control |

## Key Findings

- **σ* × k ≈ 1.79** (asymptote from fit)
- **Z_6 works** - supports "not prime-power" rule
- **CRT** requires coprime factors

## Open Questions

1. Test k=37,41,43 for asymptote confirmation
2. Random LUT control to test universality
3. CRT boundary conditions (Z_2×Z_4→Z_8, etc.)

## Structure

```
experiments/valid/   - Reproducible results
experiments/invalid/ - Known issues
experiments/legacy/  - Historical
results/             - JSON outputs
analysis/            - Debug/measure
scratch/             - Exploratory
```

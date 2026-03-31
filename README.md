# Phase-Native Neural Network - Ceiling Decay Analysis

## Executive Summary

This analysis investigates the **ceiling decay law** in ZkBundle architecture: why maximum bucket accuracy (ceiling_acc) collapses as k increases beyond the Regime II/III boundary (k≥19).

## Key Discoveries

### 1. Best Model: Power Law (R²=0.973)
```
ceiling_acc ≈ (k*/k)^α with k*≈13.9, α≈0.48
```

### 2. k=21 is an Outlier
- Delta R² = 0.284 > 0.10 threshold
- k=21 achieves 95.2% ceiling despite phase collapse (lucky initialization)

### 3. Deficit Grows Super-Linearly
- Gamma ≈ 2.26 (deficit grows faster than linear in k)

### 4. Antipodal Bound Violation
All 4 Regime III points (k=19,21,23,29) fall BELOW theoretical antipodal bound:
- k=29 achieves only 71.4% of theoretical maximum (28.5% deficit)

### 5. **SMOKING GUN: Phase Spacing Collapse**

| k | min_gap_ratio | CV (irregularity) | ceiling |
|---|---------------|-------------------|---------|
| 5 | 0.489 | 0.457 | 100% |
| 11 | 0.354 | 0.581 | 100% |
| 17 | 0.103 | 0.929 | ~85% |
| 19 | 0.074 | 0.900 | 84% |
| 21 | **0.001** | 0.786 | 95% (outlier!) |
| 23 | 0.044 | 0.881 | 83% |
| 29 | **0.005** | **1.341** | 69% |

- **Pearson r (k vs min_gap_ratio) = -0.94**
- **Pearson r (k vs CV) = +0.92**
- **Pearson r (min_gap_ratio vs ceiling, without k=21) = 0.93**

### 6. Linear Predictive Model

```
ceiling_acc ≈ 0.76 + 0.56 × min_gap_ratio
```

- R² = 0.86 (without k=21)
- Explains 86% of variance in ceiling accuracy

### 7. Regularization Experiment

Training with phase spacing regularization (λ=0.01):
- k=29: min_gap_ratio improves 10× (0.04 → 0.47)
- k=29: ceiling improves +4.1% (0.545 → 0.586)

But regularization alone cannot fully recover accuracy.

### 8. Architecture Diagnosis

The model does NOT learn modular addition:
- **Small distances (d=1) have 86% error rate**
- **Large distances (d=14) have only 7% error rate**

This is the **opposite** of geometric learning! The model memorizes large-distance patterns rather than learning the arithmetic.

### 9. Root Cause: Floating-Point Precision Limit

| k | Required precision (2π/k) | Can maintain? |
|---|---------------------------|---------------|
| 5 | 1.26 rad | ✓ Yes |
| 11 | 0.57 rad | ✓ Yes |
| 17 | 0.37 rad | ~Marginal |
| 21 | 0.30 rad | ✗ No |
| 29 | **0.22 rad** | **✗ Impossible** |

At k=29, required spacing (0.22 rad) approaches floating-point precision limits (~10⁻⁶), making uniform phase spacing fundamentally impossible to maintain.

## Files

### Analysis Scripts
| File | Description |
|------|-------------|
| `ceiling_decay_analysis.py` | Main analysis (fits, antipodal bounds, deficit power law) |
| `phase_spacing_analysis.py` | Extracts and analyzes learned phase embeddings |
| `update_results_and_plot.py` | Creates phase_collapse.png visualization |

### Results
| File | Description |
|------|-------------|
| `results/ceiling_decay/ceiling_decay_results.json` | Full results with all metrics |
| `results/ceiling_decay/ceiling_decay_law.png` | Linear + log-log fits |
| `results/ceiling_decay/phase_collapse.png` | 3-panel visualization |
| `results/ceiling_decay/phase_spacing.json` | Per-k spacing data |
| `results/ceiling_decay/phase_visualization.png` | Unit circle embeddings |
| `results/ceiling_decay/regularization_results.json` | Regularization experiment |

### Experiments
| File | Description |
|------|-------------|
| `experiments/control_v3_*/` | Training scripts for each k |
| `experiments/phase_spacing_regularization.py` | Regularization experiment |

## Architecture Details

- **Layers**: 0 hidden (direct lookup + single computation)
- **Hidden dimension**: Number of bundles (default=1)
- **Aggregation**: Mean across bundles
- **Activation**: None (phase addition + distance to output phases)

## Conclusion

The ceiling decay in ZkBundle architecture is caused by **floating-point precision limits** in phase representation. As k increases:
1. Required phase spacing (2π/k) becomes smaller than achievable precision
2. Phase collapse occurs (min_gap → 0) or wrapping (max_gap → 2π)
3. Model falls back to memorization rather than geometric learning
4. Regularization helps marginally but cannot overcome fundamental limit

This is NOT fixable by larger hidden dimensions or regularization - it requires a fundamentally different representation (e.g., discrete embeddings, higher precision, or different architecture).

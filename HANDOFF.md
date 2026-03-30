# PHASE-NATIVE LLM — UPDATED HANDOFF
**Last Updated:** March 30, 2026

---

## WHAT WAS BUILT

A neural network architecture that encodes computation in geometric phase angles (holonomy) rather than scalar activations.

---

## CONTROL EXPERIMENT v3 (Random LUT)

### Purpose
Test whether ZkBundle architecture requires GROUP STRUCTURE or can learn arbitrary deterministic patterns. Uses random lookup table as ground truth.

### Architecture: ZkBundleSimpleScaled
```python
class ZkBundleSimpleScaled(nn.Module):
    def __init__(self, k):
        self.k = k
        self.input_phases = nn.Parameter(2 * math.pi * torch.arange(k) / k)
        self.output_phases = nn.Parameter(2 * math.pi * torch.arange(k) / k)
    
    def forward(self, x1, x2):
        p1 = self.input_phases[x1]
        p2 = self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        # ... distance computation
```

### Experiments Completed

| Step | File | Description |
|------|------|-------------|
| 1 | step1_mul_mod_k.py | Fixed variance issue |
| 2 | step2_ceiling_test.py | Verified phase arithmetic at all scales |
| 3 | step3_heatmap.py | DIAGONAL_FAILURE pattern discovered |
| 4 | step4_phase_resolution_scaling.py | k=5,7,11 with sigmoid fits |
| 5 | step5_k13_k17_scaling.py | Extended to k=13,17 |
| 6 | step6_normalization_fix.py | Fixed normalization bug |
| 7 | step7_k13_k17_retrain.py | Retrained with consistent normalization |

### Final Results (Wrapped Normalization)

| k | theta | beta | R² |
|---|-------|------|-----|
| 5 | 0.069 | 0.168 | 0.999 |
| 7 | 0.134 | 0.155 | 0.991 |
| 11 | 0.141 | 0.148 | 0.902 |
| 13 | 0.318 | 0.168 | 0.974 |
| 17 | 0.452 | 0.264 | 0.960 |

### Scaling Law Fits
- **Power law**: k^1.72, R²=0.92 (WINNER)
- **Constant**: R²=0.00 (ruled out)
- **Beta**: NOT universal (std=0.042 > 0.015 threshold)

### CRITICAL FINDING: DIAGONAL FAILURE
- Model learns "a ≠ b → pick larger" perfectly (100% on upper/lower triangles)
- Fails on diagonal (a == b) because phase addition cannot resolve equal inputs
- This is the STRUCTURED failure pattern - NOT random noise

### Normalization Bug (FIXED)
- Original: k=5,7,11 used x = d/k; k=13,17 used x = 2*min(d,k-d)/k
- Fixed: All use wrapped normalization x = 2*min(d,k-d)/k

### Architecture Settings
- Learning rate: 0.1
- Optimizer: Adam
- Epochs: 150
- Seeds: 10
- n_samples: 1000

---

## VERIFIED EXPERIMENTS (v1/v2)

### NEW-A: Phase Convergence
- **Result**: Mean input variance ratio = 0.012, output = 0.024

### NEW-B: Z_4 Generalization
- **Result**: 100% train AND test accuracy

### NEW-D: Scaling Law
- **Result**: C_∞ ≈ 1.786 ± 0.009

### NEW-E: Z_6 Composite Test
- **Result**: 100% accuracy, 10/10 pass

---

## FILE STRUCTURE

```
phase-native-llm/
├── README.md                     # Quick index
├── handoff.md                    # This file
│
├── experiments/
│   ├── control_v3/               # Random LUT control (THIS RUN)
│   │   ├── control_v3_step1_mul_mod_k.py
│   │   ├── control_v3_step2_ceiling_test.py
│   │   ├── control_v3_step3_heatmap.py
│   │   ├── control_v3_step3_analysis.py
│   │   ├── control_v3_step4_phase_resolution_scaling.py
│   │   ├── control_v3_step5_k13_k17_scaling.py
│   │   ├── control_v3_step6_normalization_fix.py
│   │   └── control_v3_step7_k13_k17_retrain.py
│   │
│   ├── valid/                   # Verified experiments
│   │   ├── phase_convergence.py
│   │   ├── z_4_generalization.py
│   │   ├── scaling_law.py
│   │   └── z_6_composite.py
│   │
│   ├── invalid/                 # Known issues
│   └── legacy/                  # Historical
│
└── results/
    └── control_v3/               # Control experiment outputs
        ├── phase_resolution_fit.json
        ├── step6_k5_7_11_corrected.json
        └── ...
```

---

## OPEN QUESTIONS

1. **Retrain k=11**: Fresh training to verify undertraining hypothesis
2. **Larger k**: Test k=23,29 to confirm scaling trend
3. **Beta universality**: More data needed

---

**END OF UPDATED HANDOFF**

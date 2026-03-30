# PHASE-NATIVE LLM — UPDATED HANDOFF
**Last Updated:** March 29, 2026

---

## WHAT WAS BUILT

A neural network architecture that encodes computation in geometric phase angles (holonomy) rather than scalar activations.

---

## CORE ARCHITECTURE

### ZkBundle (for Z_k modular arithmetic)
```python
class ZkBundle(nn.Module):
    def __init__(self, k):
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward(self, x1, x2):
        p1 = self.input_phases[x1]
        p2 = self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists
```

---

## COMPLETED EXPERIMENTS

### NEW-A: Phase Convergence ✅
- **Method**: Train Z_7 with 20 seeds, measure phase spacing variance
- **Result**: Mean input variance ratio = 0.012, output = 0.024
- **Conclusion**: Phases converge uniformly to 2πj/k ± offset

### NEW-B: Z_4 Generalization ✅
- **Method**: Train on 12/16 pairs, test on 4 held-out pairs
- **Result**: 100% train AND test accuracy
- **Conclusion**: Single network learns Z_4 addition. Different from CRT composition test.

### NEW-C: Non-Group Control ⚠️ (INVALID - See Q3)
- **Method**: Train on max(a,b) mod k (no group structure)
- **Results**: See table above (ratio increases with k)
- **Finding**: acc × k ≈ 1.09 + 0.605 × k, acc → ~60% at large k
- **CONCLUSION (INVALID)**: max(a,b) mod k = max(a,b) for inputs in {0..k-1}
- The mod is VACUOUS - tests lattice structure, NOT non-group
- **RETIRED** - needs random LUT control (see Q3)

### NEW-D: Scaling Law ✅
- **Method**: k = 3,5,7,11,13,17,19,23,29, 50 seeds each, 200-point sigma grid
- **Results**:
  | k | C(k) = σ*×k | std |
  |---|-------------|-----|
  | 3 | 1.7126 | 0.034 |
  | 5 | 1.7206 | 0.036 |
  | 7 | 1.7321 | 0.039 |
  | 11 | 1.7379 | 0.050 |
  | 13 | 1.7651 | 0.050 |
  | 17 | 1.7701 | 0.064 |
  | 19 | 1.7663 | 0.058 |
  | 23 | 1.7984 | 0.057 |
  | 29 | 1.7925 | 0.067 |
- **Curve Fit**:
  - Model A (constant): C_∞ = 1.755 ± 0.010, RMSE = 0.029
  - Model B (C_∞ - D/k): C_∞ = 1.786 ± 0.009, D = 0.268 ± 0.062, RMSE = 0.015
- **AIC**: ΔAIC = -9.8 → Model B favored
- **Note**: Original H2 (C_∞=1.944, D=0.623) is WRONG - retired

### NEW-E: Z_6 Composite Test ✅
- **Method**: Train Z_6 (6 = 2×3, composite, NOT prime power), 10 seeds
- **Result**: 100% accuracy, 10/10 pass
- **Conclusion**: Supports "not prime-power" rule for single networks

---

## EXISTING RESULTS (Don't Re-run)

- **CRT Composition**: Z_3×Z_5→Z_15 = 100%, Z_2×Z_2→Z_4 = 0%
- **experiment_1_results.json**: σ*×k data (50 seeds, k=3,5,7,11)
- **experiment_b_results.json**: CRT verification

---

## HYPERPARAMETERS

- Learning rate: 0.1
- Optimizer: Adam
- Epochs: 150
- n_samples: 1000

---

## OPEN QUESTIONS (For Future Agents)

### OPEN Q1: True Asymptote
- C_∞ ≈ 1.786 ± 0.009 from fit
- ln(6) = 1.7918 → within error bars! Coincidence?
- **Action**: Test k = 37, 41, 43 to see if C(k) continues growing or plateaus

### OPEN Q2: Model A vs Model B
- AIC favors Model B but only 9 data points, 2 params
- **Action**: More data (k > 29) would clarify

### OPEN Q3: NEW-C Interpretation (CORRECTION)
- **CRITICAL**: max(a,b) mod k = max(a,b) for inputs in {0,...,k-1}
- The mod is VACUOUS - no difference between operations!
- NEW-C tests lattice structure, NOT absence of structure
- **NOT a valid non-group control** - retire this finding

### What a REAL control looks like:
```python
# Option 3: Random lookup table (gold standard)
random.seed(42)
f = { (a,b): random.randint(0,k-1) for a in range(k) for b in range(k) }
label = f[(a.item(), b.item())]
```
- Pre-generate random mapping f(a,b) → {0,...,k-1}
- Same table used for all training/test
- If network learns random LUT above chance → architecture is universal approximator
- **This is the actual control to run**

### OPEN Q4: CRT Composition Boundary
- Z_2×Z_2→Z_4 = 0% (fails)
- Single Z_4 network = 100% (succeeds)
- What exactly causes CRT failure?
  - gcd(k1, k2) > 1? 
  - Non-cyclic group structure?
  - Something else?
- **Action**: Test Z_2×Z_4→Z_8, Z_3×Z_3→Z_9, Z_4×Z_5→Z_20

---

## RETIRED HYPOTHESES

- ❌ C_∞ = 1.944 (not supported by data)
- ❌ D = 0.623 (not supported by data)  
- ❌ σ*×k = constant at 1.82 (midpoint, not asymptote)

---

## FILE STRUCTURE

```
phase-native-llm/
├── minimal-handoff.md      # Quick summary
├── handoff.md              # This file - detailed guide
├── README.md               # Quick index (create)
│
├── experiments/
│   ├── valid/              # Verified experiments
│   │   ├── phase_convergence.py
│   │   ├── z_4_generalization.py
│   │   ├── scaling_law.py
│   │   └── z_6_composite.py
│   │
│   ├── invalid/            # Known issues
│   │   └── nongroup_control_INVALID.py
│   │
│   └── legacy/            # Historical experiments
│       ├── experiment_1_scaling_law.py
│       ├── experiment_2_*.py
│       ├── experiment_3_*.py
│       ├── experiment_4_*.py
│       ├── experiment_a_*.py
│       ├── experiment_b_crt.py
│       ├── experiment_c_*.py
│       └── crt_verified.py
│
├── results/
│   ├── valid/              # Verified results
│   └── legacy/            # Historical results
│
├── analysis/               # Debug, measure, noise experiments
│   ├── debug_*.py
│   ├── measure_*.py
│   └── zk_*_noise.py
│
└── scratch/                # Exploratory / unverified
    ├── mnist_experiment.py
    ├── critical_*.py
    └── phase_transition*.py
```

### Folder Descriptions:
- **experiments/valid/** - Reproducible, verified results (use these)
- **experiments/invalid/** - Known issues, do not rely on
- **experiments/legacy/** - Historical context, may have bugs
- **analysis/** - Debug scripts and measurement tools
- **scratch/** - Exploratory code, not verified

---

**END OF UPDATED HANDOFF**

# PHASE-NATIVE LLM — HANDOFF DOCUMENT
**Last Updated:** March 29, 2026
**Status:** Ready for bulletproof suite

---

## WHAT WAS BUILT

A neural network architecture that encodes computation in geometric phase angles (holonomy) rather than scalar activations.

---

## CORE ARCHITECTURE

### HolonomyChainBundle (for binary parity)
```python
class HolonomyChainBundle(nn.Module):
    def __init__(self, n_bits: int):
        self.bit_phases = nn.Parameter(torch.ones(n_bits) * math.pi)
        self.phi_0 = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.ones(n_bits) * math.pi)
    
    def forward(self, inputs):
        phi = self.phi_0.expand(batch_size)
        for i in range(self.n_bits):
            phi = phi + inputs[:, i] * self.bit_phases[i]
        output = (1.0 - torch.cos(phi)) / 2.0
        return output
```

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

## CONFIRMED RESULTS

### 1. Scaling Law (MAIN) ⚠️
- σ* × k ≈ 1.82 (initial approximation, INCOMPLETE)
- **Data shows GROWTH with k:**
  - k=3:  σ* = 0.579, σ*×k = 1.736
  - k=5:  σ* = 0.357, σ*×k = 1.786
  - k=7:  σ* = 0.258, σ*×k = 1.803
  - k=11: σ* = 0.166, σ*×k = 1.823
  - k=13: σ* = 0.145, σ*×k = 1.879
  - k=17: σ* = 0.112, σ*×k = 1.907
- **Empirical fit:** σ* × k = 1.944 - 0.623/k
- **As k→∞:** approaches ~1.944 (NOT π/√3 = 1.814)

### 2. CRT Composition (STRONGEST RESULT)
- gcd=1 pairs: Z_3×Z_5→Z_15 = **100%** ✓
- gcd>1 pairs: Z_2×Z_2→Z_4 = **0%** ✓
- This is a PREDICTION that came true: CRT says it should fail, it fails

---

## FAILED / DEAD END EXPERIMENTS

### Holonomy Closure (Experiment 4)
- **FAILED CONTROL**: untrained network achieves holonomy = 1.0
- Measurement is vacuous - tests associativity of addition, always true
- Divergence numbers (0.569 at σ=0.35) are ALSO invalid because they used the vacuous metric
- The raw accuracy degradation curve is real data - keep but don't cite the holonomy column

### Curvature Q Metric
- Always Q = 2/k by construction - not a result
- Relabeled as DEAD END

---

## THE BULLETPROOF SUITE (PENDING)

### EXP 1: σ* × k = 1.82 ✓ DONE
- Status: REAL
- Closes: baseline law exists

### EXP 2: CRT Composition ✓ DONE  
- Status: REAL
- Closes: hatch 1 (not lookup table)

### EXP NEW-A: Phase Convergence Across Seeds
- **Method**: Train k=7 with 20 random seeds. Measure variance of phase spacing. Compare to 2π/k prediction.
- **Expected**: All seeds converge to 2πj/k ± small offset
- **Closes**: hatch 2 (phases are meaningful, not arbitrary)

### EXP NEW-B: Failure Case Z_4 ≠ Z_2×Z_2
- **Method**: Already done - Z_2×Z_2 → Z_4 = 0% closure
- **Status**: Should reconfirm with fresh run
- **Closes**: hatch 4 (CRT is real, not artifact)

### EXP NEW-C: Non-Group Control (MAX mod k)
- **Method**: Train on max(a,b) mod k (no group structure). Measure phase uniformity.
- **Expected**: Phases should be LOW/random (no geometric emergence)
- **Closes**: hatch 5 (structure is group-specific)

### EXP NEW-D: Identify the Constant ⚠️

**⚠️ TWO HYPOTHESES TO TEST:**

**H1:** σ*×k = constant
- Candidate: π/√3 ≈ 1.814
- Current data: 1.8224 ± 0.0572
- Delta = 0.0086 < σ (cannot reject, cannot confirm)

**H2:** σ*×k = A - B/k (convergent series)
- Empirical fit: A ≈ 1.944, B ≈ 0.623
- Asymptote ≈ 1.944 (ruled out if H1 is true)
- **This is distinguishable from H1 with k up to 29**

**Required:**
- Run k = 3, 5, 7, 11, 13, 17, 19, 23, 29
- High-precision σ* measurements (100+ seeds each)
- Fit both models, compare residuals

**If H2 wins:** The "law" is σ* = A/k - B/k²
- Asymptote ~1.944, NOT π/√3
- Paper title changes: "π/(k√3)" → "A/k - B/k²"

---

## PRIORITY ORDER

1. **NEW-B first** - Reconfirm Z_4 failure (strongest single experiment)
2. **NEW-A next** - Phase convergence (quick, bolsters representation claim)
3. **NEW-C next** - Non-group control (most important for reviewers)
4. **NEW-D last** - Identify constant (needs tighter measurements first)

---

## HYPERPARAMETERS

- Learning rate: 0.1
- Optimizer: Adam
- Epochs: 150-200
- n_samples: 1000-2000

---

## FILE STRUCTURE

```
phase-native-llm/
├── HANDOFF.md
├── experiment_crt_verified.py       # CRT composition (DONE)
├── experiment_1_scaling_law.py      # sigma* x k (DONE)
├── experiment_3_input_structure.py # Training noise structure (DONE)
├── experiment_4_holonomy_accuracy.py # Divergence (done but holonomy broken)
├── experiment_3_input_structure.json
├── experiment_4_holonomy_accuracy.json
├── experiment_crt_verified.json
└── [NEW EXPERIMENTS TO CREATE]
    ├── experiment_new_a_seed_convergence.py
    ├── experiment_new_c_nongroup_control.py
    └── experiment_new_d_identify_constant.py
```

---

## CRT FUNCTION (VERIFIED)

```python
def crt(a1, m1, a2, m2):
    """Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2)"""
    import math
    if math.gcd(m1, m2) != 1:
        return None
    inv_m1 = pow(m1, -1, m2)
    inv_m2 = pow(m2, -1, m1)
    M = m1 * m2
    return (a1 * m2 * inv_m2 + a2 * m1 * inv_m1) % M
```

---

## KEY EQUATIONS

- σ* × k ≈ 1.8224 (empirical, GROWING with k)
- σ* × k = 1.944 - 0.623/k (empirical fit, asymptote ~1.944)
- π/√3 ≈ 1.814 (ruled out as asymptote if H2 confirmed)
- Q = 2/k (curvature - DEAD END, by construction)
- Phase uniformity = 1 - std(diffs)/mean(diffs)

---

## RELATED WORK

### 1. "Grokking Modular Arithmetic" (2023)
   arxiv.org/abs/2301.02679
   - Networks solving Z_k addition learn Fourier features. Analytic weight expressions derived.
   - RELEVANCE: Our phase convergence (NEW-A) should reproduce their Fourier finding in phase language. If they match → same phenomenon, different frame. If they don't → something new.

### 2. "Fourier Circuits in Neural Networks and Transformers" (Li et al., AISTATS 2025)
   proceedings.mlr.press/v258/li25b.html
   - Margin maximization → Fourier features for Z_k. Transformers learn "integer rotations around a circle"
   - RELEVANCE: **THIS IS US.** They describe the same geometric algorithm we built explicitly. Our contribution: made it the architecture, not just an emergent property.

### 3. "Fiber Bundle Networks" (2024)
   arxiv.org/abs/2512.01151
   - FiberNet: classification via fiber bundle geometry.
   - RELEVANCE: Independent convergence on same idea. Cite as parallel work. Do NOT present as derivative.

### 4. ModuloNET (IACR 2021)
   eprint.iacr.org/2021/1437.pdf
   - Modular arithmetic in NNs for cryptographic masking.
   - RELEVANCE: Demonstrates modular arithmetic is compatible with NN training. Different motivation (security, not geometry) but validates the arithmetic approach.

**KEY FRAMING**: Li et al. (2025) describe our architecture as an emergent phenomenon in standard transformers. Our contribution: made it explicit and controllable. That framing survives peer review.

---

## IMPORTANT NOTES

- IDE LSP errors about torch are FALSE POSITIVE - PyTorch works fine
- Unicode issues on Windows - use ASCII in print statements
- CUDA not available - running on CPU
- All experiments complete in <10 minutes

---

**END OF HANDOFF**

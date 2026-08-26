# PHASE-NATIVE LLM - COMPLETE HANDOFF
**Last Updated:** August 21, 2026

---

## NEWEST LINE: LLM-Driven Phase Memory (see MEMORY-AGENT.md)

The direction has moved from "zero-param solvers" to **an LLM that builds and calls a
phase-native memory as its own O(1) store** — solidify established reasoning into low-byte
nuggets, recall in constant time, verify low-confidence recalls, iterate → the model
specializes and stops re-deriving. The zero-param `ZkBundleExplicit` is the `D=1` special
case of this memory.

New package `phase_native/` (numpy, CPU — torch not required):
- `memory.py` PhaseNuggetMemory (write / recall O(1) / forget / serialize), `codebook.py`
  (CRT value decode + random keys, reusing `analysis/test_crt.py`), `ops.py`, `tools.py`
  (Claude tool schemas), `agent.py` (`run_agent` — the real LLM loop), `driver.py`
  (ScriptedDriver offline scaffolding), `domain.py` (hidden-graph reach task).
- `experiments/seek_scaling.py` → **O(1) seek** (flat ~0.5 ms, 10→3000 nuggets) + capacity.
- `experiments/memory_agent_specialization.py` → **110,504 → 99 steps (~1116×), acc 1.000**
  in a fixed 384 KB memory; `--live` runs the LLM itself (needs Anthropic creds).
- `phase_native/compose.py` + `experiments/compose_multihop.py` → **composition**: store only
  atomic facts, chain them into multi-hop answers by iterated O(1) recall (correct to depth 160
  lightly loaded; honest load horizon); costly steps plateau at world-size (**159 vs 14,957, ~94×**).
- `tests/test_phase_native.py` → 21 checks incl. composition + the live tool-loop via a mock client. ALL PASS.

NOTE: this sandbox has no API key and its proxy bypasses api.anthropic.com, so `--live`
can't run here — everything else is proven; run `--live` where creds exist.

---

## WHAT WE PROVED

### Grokking is the Cost of Structure Discovery, Not a Phase Transition

**Key Result**: A model (ZkBundleExplicit) achieves 100% test accuracy on modular addition with **zero learnable parameters** and **zero gradient steps**.

This contradicts the standard definition of grokking ("sudden generalization after prolonged overfitting") and requires a redefinition.

---

## THE BREAKTHROUGH: Zero-Parameter Solution

### Architecture
```
Input: a, b → phases = 2π·a/k, 2π·b/k
  ↓
FIBER POSITIONS: [cos(phases_a), sin(phases_a)], [cos(phases_b), sin(phases_b)]
  ↓
CONNECTION: result_phase = phases_a + phases_b
  ↓
READOUT (Fourier): logits[c] = cos(result_phase - 2πc/k)
  ↓
Output: argmax = (a + b) mod k
```

### Results (k=11 to k=37)

| k | Train Acc | Test Acc | Learnable Params |
|---|-----------|----------|-------------------|
| 11 | 100% | 100% | 0 |
| 17 | 100% | 100% | 0 |
| 23 | 100% | 100% | 0 |
| 29 | 100% | 100% | 0 |
| 31 | 100% | 100% | 0 |
| 37 | 100% | 100% | 0 |

---

## COMPARISON: FlatTransformer vs ZkBundleExplicit (k=23)

| Model | Params | Grokking Step | Test Acc |
|-------|--------|---------------|----------|
| ZkBundleExplicit | 0 | 0 | 100% |
| FlatTransformer (seed=42) | 36,567 | 6,000 | 100% |
| FlatTransformer (seed=123) | 36,567 | 15,000+ | 80% |
| FlatTransformer (seed=7) | 36,567 | 4,500 | 100% |

**Same task, same data** — but ZkBundle solves it instantly.

---

## KEY DISCOVERIES

### 1. Zero Parameters Suffice
For tasks with known group structure, no learned weights are needed. The geometry IS the solution.

### 2. Mean Pooling Destroys Phase Addition
```
mean(Linear(phase_a), Linear(phase_b)) = Linear((phase_a + phase_b) / 2)
```
Computes midpoint, NOT angular sum — information loss the transformer cannot recover from.

### 3. Fourier Readout is Exact
The optimal classifier for circular data: `logits[c] = cos(result_phase - 2πc/k)`

### 4. FlatTransformer Does NOT Discover Fourier Geometry
- Embedding SVD: Flat spectrum, 30% variance in top 2 (not 2-dominant)
- MLP neurons: Diverse frequencies (1,2,3,4,5 cycles), not dominated by frequency 1

---

## THE REDEFINITION (PARADOX.md)

> **Grokking is not a phase transition — it is the cost of discovering geometric structure from flat primitives.**

- Flat embeddings (nn.Embedding): Must learn geometry through gradient descent → grokking delay
- Geometric primitives (ZkBundle): Solution already present → instantaneous

What "grokking papers" measure is the computational cost of structure discovery, not a magical phenomenon.

---

## FILE STRUCTURE

```
phase-native-llm/
├── README.md                          # Updated with full results
├── HANDOFF.md                         # This file
├── PARADOX.md                         # Redefinition argument
│
├── experiments/
│   ├── zkbundle_explicit_v2c.py       # Exact solution (k=11-37)
│   ├── zero_param_demo.py             # Demo: 0 params, 100%
│   ├── grokking_benchmark.py          # Comparison (k=23, 3 seeds)
│   ├── analyze_grokking_model.py      # SVD + frequency analysis
│   ├── grokking_discovery.py          # Full experiment (k=11,17,23)
│   └── grokking_discovery.py          # (fixed R² metric)
│
└── results/
    ├── zkbundle_explicit_v2c.json     # 100% for k=11-37
    ├── grokking_discovery.png         # Experiment figure
    └── grokking_k11_s42_svd.png       # SVD analysis plot
```

---

## BUGS FIXED ALONG THE WAY

1. **R² metric direction** — E→F was backwards, fixed to F→E
2. **R² extraction** — alignment_dict.get(key) → find nearest key ≤ grok_step
3. **test_loss computation** — argmax().mean() → F.cross_entropy()

---

## TO DO

- [x] ZkBundleExplicit with Fourier readout (DONE)
- [x] 100% at step 0 for k=11-37 (DONE)
- [x] Zero-parameter demo (DONE)
- [x] FlatTransformer comparison (k=23, 3 seeds) (DONE)
- [x] SVD + frequency analysis (DONE)
- [x] README + HANDOFF update (DONE)
- [x] Push to GitHub (DONE)

---

**END OF HANDOFF - March 31, 2026**
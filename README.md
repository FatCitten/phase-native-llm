# Phase-Native LLM - Grokking Delay = Primitive Mismatch

## Executive Summary (March 31, 2026)

This directory contains experiments proving that **grokking is the cost of structure discovery, not a magical phase transition**.

---

## KEY RESULT: Exact Solution with Zero Parameters

### ZkBundleExplicit - Fourier Readout

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

**Result: 100% accuracy at step 0 with ZERO learnable parameters!**

| k | Train Accuracy | Test Accuracy | Learnable Params |
|---|----------------|---------------|------------------|
| 11 | 100.00% | 100.00% | 0 |
| 17 | 100.00% | 100.00% | 0 |
| 23 | 100.00% | 100.00% | 0 |
| 29 | 100.00% | 100.00% | 0 |
| 31 | 100.00% | 100.00% | 0 |
| 37 | 100.00% | 100.00% | 0 |

---

## COMPARISON: FlatTransformer vs ZkBundleExplicit (k=23)

| Model | Learnable Params | Grokking Step | Test Accuracy |
|-------|-----------------|---------------|---------------|
| ZkBundleExplicit | 0 | 0 | 100% |
| FlatTransformer (seed=42) | 36,567 | 6,000 | 100% |
| FlatTransformer (seed=123) | 36,567 | 15,000+ | 80% |
| FlatTransformer (seed=7) | 36,567 | 4,500 | 100% |

**Same task, same data** — but ZkBundle solves it instantly, FlatTransformer needs thousands of gradient steps.

---

## THE PARADOX

Grokking papers define grokking as "sudden generalization after prolonged overfitting." But our zero-parameter solution achieves 100% generalization at step 0.

**New definition**: Grokking is *not* a phase transition — it is the cost of discovering geometric structure from flat primitives.

- Flat embeddings (nn.Embedding): Must learn geometry through gradient descent → grokking delay
- Geometric primitives (ZkBundle): Solution already present → instantaneous generalization

See **PARADOX.md** for the full argument.

---

## Key Discoveries

1. **Zero parameters suffice** — For tasks with known group structure, no learned weights are needed
2. **Mean pooling destroys phase addition** — `mean(Linear(phase_a), Linear(phase_b)) = Linear((phase_a + phase_b)/2)` — computes midpoint, not angular sum
3. **Fourier readout is exact** — The optimal classifier for circular data
4. **FlatTransformer does NOT discover Fourier geometry** — SVD shows flat spectrum, MLP neurons use diverse frequencies

---

## File Structure

```
phase-native-llm/
├── README.md                          # This file
├── HANDOFF.md                         # Detailed handoff
├── PARADOX.md                         # Redefinition of grokking
│
├── experiments/
│   ├── zkbundle_explicit_v2c.py       # Exact solution (k=11-37)
│   ├── zero_param_demo.py             # Demo: 0 params, 100% acc
│   ├── grokking_benchmark.py          # Comparison (k=23, 3 seeds)
│   ├── analyze_grokking_model.py      # SVD + frequency analysis
│   └── grokking_discovery.py          # Full experiment (k=11,17,23)
│
└── results/
    ├── zkbundle_explicit_v2c.json     # 100% for k=11-37
    └── grokking_discovery.png         # Experiment figure
```

---

## How to Run

```bash
# Zero-parameter solution (k=11-37)
python experiments/zkbundle_explicit_v2c.py

# Demo: 0 params, 100% accuracy
python experiments/zero_param_demo.py

# Comparison with FlatTransformer (k=23)
python experiments/grokking_benchmark.py
```

---

## Scientific Implications

1. **Grokking measures structure discovery cost** — Not a magical phase transition
2. **The connection operation is the knowledge** — Hardcode angle addition, readout is trivial
3. **Zero parameters possible** — For known group structure, pure geometry suffices
4. **Floating-point precision limits** — For large k, even geometric solutions degrade
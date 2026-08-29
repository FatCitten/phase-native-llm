# Phase-Native LLM - Grokking Delay = Primitive Mismatch

## Executive Summary (March 31, 2026)

This directory contains experiments proving that **grokking is the cost of structure discovery, not a magical phase transition**.

---

## Latest: LLM-Driven Phase Memory — an O(1) "self-agent" (see MEMORY-AGENT.md)

The zero-parameter result below is the `D=1` special case of a general idea: give an **LLM a
phase-native associative memory it drives itself** — it decides what established reasoning to
solidify into low-byte nuggets, writes them, and recalls them in **constant time regardless
of how much it has stored** (O(1) seek), verifying low-confidence recalls. "Blindness is the
enemy": the LLM's judgment is the control surface, not a blind pipeline.

Proven here (CPU, no API):
- **O(1) seek** — phase recall stays flat (~0.13 ms) from 10 to 3000 stored nuggets while a
  naive scan grows linearly (~17× slower by 3000) (`experiments/seek_scaling.py`).
- **Specialization** — an agent answering `reach(s,k)` on a hidden graph uses **110,504 → 99
  steps (~1116× less compute) at unchanged 100% accuracy** once it solidifies jump nuggets
  into a fixed **384 KB** memory (`experiments/memory_agent_specialization.py`).
- **Composition** — the LLM stores only *atomic* facts and chains them into multi-hop answers
  by iterated O(1) recall (`recall_chain`): correct to depth 160 when lightly loaded, with an
  honest load-dependent horizon; costly steps plateau at ≤ world-size — **159 vs 14,957 (~94×)**
  over a query stream (`experiments/compose_multihop.py`).
- **Live-verified** — the real LLM driver ran on the API (Haiku 4.5, 8/8 correct): on a
  recurring start node it paid a few steps once, then answered every later query — including a
  13-hop one — in **0 steps** by O(1) recall (`results/memory_agent_live.json`). Also runs on
  open models via Ollama (`experiments/memory_agent_ollama.py`).

```bash
python tests/test_phase_native.py && python experiments/seek_scaling.py \
  && python experiments/memory_agent_specialization.py
```

---

## Also: synaptic pruning → iterative consolidation (the developmental loop)

Two numpy toys (CPU, no API) test the biological thesis — *solidifying memory by pruning
amplifies capability, and standard training is blind to the structure it builds*:

- **Blindness + amplification** (`experiments/synaptic_pruning.py`). A trained MLP over-produces
  synapses: on a teacher-student task **50% are removable with test accuracy essentially
  unchanged**, and capability-per-surviving-synapse rises up to **~40×** as we prune to the
  structure that matters. On modular addition the same net reaches **100% train / 0% test** — the
  objective is perfectly satisfied while the model understands nothing (the loss sees only the
  landing spot). A winning ticket (init + solidified mask) relearns faster than a random subnet of
  equal sparsity: *the specific structure is the memory* (Lottery Ticket Hypothesis, Frankle &
  Carbin 2018).
- **Iterative consolidation** (`experiments/consolidation_rounds.py`). A growing/pruning net runs
  the loop in waves — **overproduce** candidate fibers → **train** toward the objective (frozen
  base untouched) → **relational-prune the void** → **freeze** survivors as new primitives.
  Measured honestly, kill-criteria stated up front:
  - **Axioms are inviolate** — round-1 survivors are byte-identical after every later round.
  - **Structure grows outward** — new primitives establish at ever-greater *relational distance
    from the axioms* (mean **1.00 → 1.83** hops over 5 rounds, max → 2.5), where *distance = the
    gap/void of non-relation* to the established base.
  - **The void is real** — 3–12% of overproduced units and ~40% of candidate connections relate
    to nothing and are pruned every round, at no accuracy cost (blindness, re-observed).
  - **Honest cost** — on this shallow task consolidation *trades efficiency for inviolability*:
    the loop reaches 0.703 using 7,238 wires while an unconstrained net of the same width reaches
    0.727 using 3,504. Consolidation buys frozen, outward-growing structure here — not fewer
    synapses. Whether it pays off in efficiency needs a task with genuine deep reuse (flagged, not
    faked).

```bash
python experiments/synaptic_pruning.py && python experiments/consolidation_rounds.py
```

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
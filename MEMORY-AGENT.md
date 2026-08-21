# Phase-Native Memory — an O(1) "self-agent" the LLM builds and calls

## The idea

Training a model re-derives established structure over and over. The alternative this repo
has been circling: don't re-derive it — **solidify it into memory and recall it**. The
correction that shaped this module:

> **Blindness is the enemy.** A blind, deterministic distiller is not the solution. The
> **LLM itself** must guide the grokking and save the nuggets — a "self-agent" it can
> **infinitely call on and rely on as its own memory, with an O(1) seek.**

So the LLM is the sighted driver. It decides what is *established* enough to crystallize, it
**writes** nuggets into a phase-native memory, and it **reads** them back in constant time —
offloading settled knowledge instead of recomputing it. The memory is "trained" by
LLM-guided consolidation, not gradient descent. And because the LLM is in the loop, it
**verifies** what it recalls: a low-confidence recall is re-derived. Sightedness covers the
substrate's imperfection.

This generalises the zero-parameter result in `experiments/zkbundle_explicit_v2c.py`: that
model (integers→phase, angle addition = the "connection", Fourier readout) is exactly the
**D=1, single-binding special case** of the memory below.

## The substrate (Fourier Holographic Reduced Representation)

Every association lives in **one fixed-size complex vector** `M` by superposition:

```
M = sum_j  bind(key(cue_j), value(conclusion_j))          # elementwise, O(dim), no gradients
recall(cue) = decode( unbind(M, key(cue)) )               # phase subtraction, then decode
forget(cue) = M -= bind(key(cue), value(cue))             # algebraic edit
```

| piece | choice | why |
|-------|--------|-----|
| **keys** (cues) | deterministic random phase vectors from a hash of the cue string | near-orthogonal → low cross-talk; unbounded key space; never decoded |
| **values** (conclusions) | **CRT** phases over coprime moduli (`8,9,5,7`), reused from `analysis/test_crt.py` | a value *is* a small residue tuple (low-byte); decodable channel-by-channel |
| **bind / unbind** | complex multiply / conjugate-multiply = phase add / subtract | the "connection" of `zkbundle_explicit_v2c.py`, lifted to a vector of circles |
| **decode** | matched-filter cleanup (default) or CRT residues | see below |

## Why the seek is O(1) (and the honest caveat)

Retrieval is `unbind` (O(dim)) + decode. **`unbind` is constant in N — the number of stored
nuggets** — because everything is superposed into one fixed vector; you never scan what you
stored. That is the "does not slow down as it fills up" property the LLM relies on.

- **CRT decode** is O(dim), constant in **both** N and the value vocabulary V → strict O(1).
- **cleanup decode** (matched filter) is O(V), higher capacity, and its normalized score is
  a real **confidence** (≈1.0 clean hit, ≈0.3 noise) — which is what lets the LLM verify.
  The agent uses cleanup; V is the answer alphabet (small, fixed by the task), not N.

Measured (`experiments/seek_scaling.py`, dim=4096):

| nuggets N | phase seek | naive scan |
|-----------|-----------|-----------|
| 10 | 0.51 ms | 0.02 ms |
| 300 | 0.49 ms | 0.30 ms |
| 3000 | **0.53 ms** | **5.39 ms** |

Phase seek is flat as N grows 300×; a naive store-and-scan grows linearly and is 10× slower
by N=3000. **Caveat:** O(1) *time*, **bounded capacity** — recall fidelity degrades as load
approaches capacity (tunable via dim), so the LLM verifies and re-derives low-confidence
recalls. Capacity is measured, not hidden.

## The consolidation policy (what the LLM does)

1. **Recall first** — `memory_recall(cue)` before doing work.
2. **Verify** — trust confidence ≥ 0.45; otherwise treat as a miss.
3. **Compute on miss** — derive the result (here: binary-lifting jumps, base steps only when needed).
4. **Solidify** — `memory_write(cue, conclusion)` so it is an O(1) recall next time.

## The demonstrator: compute-to-answer collapses as structure is solidified

Task (`phase_native/domain.py`): answer `reach(s, k)` on a **hidden** graph traversable only
via a costly `step`. Queries share sub-structure (binary-lifting jumps), so solidified jumps
turn re-derivation into O(1) recall. `experiments/memory_agent_specialization.py`, 200
queries:

| | total compute (steps) | accuracy |
|---|---|---|
| memory **OFF** (re-derive every time) | 110,504 | 1.000 |
| memory **ON** (solidify + recall) | **99** | **1.000** |

**~1116× less compute, same accuracy** — 571 nuggets in a fixed **384 KB** vector, 1389 O(1)
recall hits. That is the specialization signal: as the LLM solidifies established results,
per-query compute falls toward zero while correctness holds.

## Running it

```bash
python tests/test_phase_native.py                          # all offline, no API
python experiments/seek_scaling.py                         # O(1) seek + capacity figure
python experiments/memory_agent_specialization.py          # offline specialization (scripted)
python experiments/memory_agent_specialization.py --live --n 8   # the LLM drives it (needs creds)
```

## The real driver vs the scaffolding

`phase_native/agent.py` (`run_agent`) is the **solution**: a manual Claude tool-use loop
(`memory_recall` / `memory_write` / `step` / `final_answer`), adaptive thinking, `usage`
captured as the compute metric. It needs Anthropic credentials.

`phase_native/driver.py` (`ScriptedDriver`) is **test scaffolding** — a deterministic
stand-in that drives the *same tools* with the *same policy*, so the mechanics and the
compute-offload effect are provable offline. It is not the solution; "blindness is the
enemy." Its tool loop is byte-for-byte the interface the LLM uses.

> **This sandbox has no Anthropic credentials, and its egress proxy bypasses
> `api.anthropic.com`, so the live LLM run cannot execute here.** Everything else — the
> substrate, the O(1) seek, capacity, the full tool loop (via a mock client in the tests),
> and the specialization effect (via the scripted driver) — is proven here. Run the `--live`
> command in an environment that has a key to see the LLM itself drive the memory.

## Honest limits

- **Bounded capacity.** Overfill the memory and confident-but-wrong recalls appear (measured
  in `seek_scaling.py`). Size dim to the working set; the LLM's verify-gate covers the edge.
- **The offline proof is a stand-in.** It proves loop mechanics + the metric, not that real
  LLM reasoning always compresses to clean cue→conclusion nuggets — the `--live` path is the
  first probe of that.
- **Not SOTA, by design.** The claims are O(1) associative recall, LLM-guided compute-offload,
  and full control (inspect / edit / forget) — each measured above.

## Files

```
phase_native/
  codebook.py   CRT value encoding (O(1) decode) + deterministic random keys + CRT
  ops.py        bind / unbind / superpose / cleanup (elementwise, no gradients)
  memory.py     PhaseNuggetMemory: write / recall(O(1)) / forget / serialize
  tools.py      Claude tool schemas + executor (memory_recall/write, step, stats)
  driver.py     Driver protocol + ScriptedDriver (offline scaffolding)
  agent.py      ClaudeDriver / run_agent — the real LLM-driven loop
  domain.py     hidden-graph reach(s,k) task with recurring sub-structure
experiments/
  seek_scaling.py                    O(1) seek + capacity/fidelity
  memory_agent_specialization.py     specialization curve (offline + --live)
tests/test_phase_native.py           self-checking suite (no API)
```

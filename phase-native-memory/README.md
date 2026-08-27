# Phase-Native Memory

**An O(1) associative memory an LLM builds and drives itself.** One fixed-size vector. No
gradients, no training, no server. The model decides what to solidify, recalls it in constant
time no matter how much it has stored, and — unlike every similarity/RAG store — *knows when it
doesn't know*.

```bash
pip install numpy        # the entire runtime dependency
python demo.py           # four proofs, computed live on CPU in ~30s
```

Everything below is produced by that command, on your machine, with no API key. Nothing is
read from a cache; if a number looks too good, the code that made it is one short file away.

---

## Why this exists

An LLM re-derives the same settled sub-results over and over — within a session, and from
scratch in the next one. Context windows forget. Vector databases grow slower as they fill and
will confidently hand back a nearest neighbour for a fact you never stored. This is a different
kind of memory, built to be driven *by the model*:

| property | what it means for the model |
|---|---|
| **O(1) seek** | recall time is flat whether you've stored 10 nuggets or 3000 — lean on it without penalty |
| **LLM-guided** | the model chooses what to crystallize and when to trust a recall — *"blindness is the enemy"* |
| **Verifiable** | every recall carries a calibrated confidence; low confidence ⇒ re-derive, don't hallucinate |
| **Compositional** | store atomic facts, chain them into multi-hop answers that were never written |
| **Editable** | inspect / overwrite / `forget` any nugget algebraically — a fixed-size vector, fully auditable |
| **Tiny & portable** | associations superpose into one complex vector; serialize the whole memory to a few hundred KB |

---

## The four proofs (`python demo.py`)

### 1. O(1) seek — recall doesn't slow down as you store more
Every association is superposed into one fixed vector; retrieval is a phase *unbind* + decode,
O(dim), independent of the number of nuggets `N`.

```
   N stored     phase seek     naive scan   speedup
       1000       305 us          1706 us      5.6x
       3000       270 us          7248 us     26.8x
```
Phase seek is flat; a naive store-and-scan grows linearly and is ~27× slower by N=3000 and
climbing. *(At tiny N the naive scan is cheaper — the phase seek has a fixed decode cost. The
invariant is the shape: flat vs. linear.)* — `experiments/seek_scaling.py`

### 2. Specialization — solidify results, stop re-deriving them
Answer `reach(s,k)` on a **hidden** graph walkable only via a costly `step`. Same agent, same
policy, same task — the only difference is whether it has the memory.

```
                compute (steps)   accuracy
   memory OFF           110,504     100.0%
   memory ON                 99     100.0%     ->  1,116x less compute
```
571 nuggets in a fixed **384 KB** vector; compute-to-answer collapses toward zero as established
structure is solidified, accuracy untouched. — `experiments/memory_agent_specialization.py`

### 3. Composition — combine atomic facts into answers you never stored
Store only one-step edges. Answer deep multi-hop queries by chaining O(1) recalls — no new
writes, no new compute.

```
   chain depth   accuracy   new steps  new writes
           160      100%           0           0
```
Depth-160 answers assembled from 64 stored facts, correctly, at zero marginal cost. Per-hop
fidelity compounds, so the reliable depth is set by memory load — an honest, measured horizon.
— `experiments/compose_multihop.py`

### 4. Lucidity — it will not lie to you
A similarity store returns a nearest match for *any* query. This one returns a calibrated
confidence and abstains.

```
   load  true recall  confabulation   known-unknown margin
     50        100%          0.0%              +0.30   [lucid]
    110         97%         13.9%              -0.01   [past capacity]
```
Inside a self-diagnosed capacity: 100% recall, **0% confabulation**. As you overfill, the
known-vs-unknown confidence margin crosses zero — the model can *see* the edge coming and
re-derive instead of trusting a bad recall. — `experiments/lucidity.py`

---

## How an LLM uses it (the whole interface)

The model is handed four tools and one policy: **recall first; trust confidence ≥ gate;
compute on a miss; solidify the result.** That's it.

```python
from phase_native import PhaseNuggetMemory

mem = PhaseNuggetMemory()            # one fixed-size vector; no training

mem.write("capital(france)", "paris")           # solidify an established result
r = mem.recall("capital(france)")               # O(1) associative seek
r.payload, r.confidence                          # -> ('paris', ~1.0)  trust it

r = mem.recall("capital(atlantis)")             # never stored
r.hit, r.confidence                              # -> (False, ~0.3)   it says "I don't know"

mem.forget("capital(france)")                    # algebraic edit, exact
```

The same four tools (`memory_recall`, `memory_write`, `step`, `memory_stats`) are exposed as
Anthropic/OpenAI tool schemas in `phase_native/tools.py`, so a live model drives the memory
itself. **Real Claude (Haiku 4.5) already did — 8/8 correct end-to-end** (`results/memory_agent_live.json`):
on a recurring start node it paid a couple of `step`s once, then answered every later query —
including a 13-hop one — in **0 steps**, purely by recall.

---

## Reproduce everything

```bash
pip install -e ".[demo]"                          # numpy + matplotlib (for the figures)

python demo.py                                    # the four proofs, in-process, ~30s
python tests/test_phase_native.py                 # 21 self-checks, no API
python experiments/seek_scaling.py                # O(1) seek + capacity figure
python experiments/memory_agent_specialization.py # 110,504 -> 99 steps
python experiments/compose_multihop.py            # multi-hop composition + horizon
python experiments/lucidity.py                    # zero-confabulation capacity
python experiments/zero_param_origin.py           # the origin: 100% at step 0, 0 params
```

Drive it with a **live** model (needs credentials):

```bash
pip install -e ".[live]"
python experiments/memory_agent_specialization.py --live --n 8    # Claude drives it
python experiments/memory_agent_ollama.py --model gpt-oss:120b    # any tool-capable open model
```

---

## How it works (one paragraph)

Every association is a **bind** — an elementwise complex multiply, i.e. per-channel phase
addition — of a *key* (a deterministic random phase vector hashed from the cue string,
near-orthogonal so bindings don't collide) and a *value* (an integer encoded as CRT phases over
small coprime moduli, decodable channel-by-channel without scanning any vocabulary). All
bindings **superpose** into one fixed vector `M`. Recall is **unbind** (multiply by the
conjugate key) then decode — O(dim), constant in how many nuggets you stored. This is the
zero-parameter modular-addition trick (`experiments/zero_param_origin.py`: put integers on a
circle, add the phases, read out by Fourier basis → 100% accuracy, 0 parameters, 0 gradient
steps) lifted from a single circle to a vector of circles. The full derivation is in
[`FOR_AN_LLM.md`](FOR_AN_LLM.md).

---

## Honest limits

- **Bounded capacity.** Overfill the vector and confident-but-wrong recalls appear. Capacity is
  *measured* (`seek_scaling.py`, `lucidity.py`), tunable via `dim`, and self-diagnosed by the
  confidence margin — the verify-gate is what covers the edge. It is O(1) *time*, bounded *space*.
- **The offline driver is a stand-in.** `ScriptedDriver` proves the loop mechanics and the
  compute-offload metric deterministically; it is explicitly *not* the solution. The `--live`
  path is the real driver, and it has been run (Haiku 4.5, 8/8).
- **Not a language model, and not SOTA-chasing.** This is an associative *substrate* an LLM
  drives. The claims are exactly the four proofs above — each one reproducible on your CPU.

---

## Repository map

```
phase_native/        the working code (numpy only)
  ops.py             bind / unbind / superpose / cleanup — elementwise, no gradients
  codebook.py        CRT value encoding (O(1) decode) + deterministic random keys
  memory.py          PhaseNuggetMemory: write / recall(O(1)) / forget / serialize
  compose.py         recall_chain / compose_reach / lifted_reach — multi-hop composition
  tools.py           memory tools as Anthropic/OpenAI schemas + executor
  domain.py          the hidden-graph reach(s,k) task with recurring sub-structure
  driver.py          ScriptedDriver — deterministic offline stand-in
  agent.py           run_agent — the real Claude-driven loop  (extra: [live])
  ollama_agent.py    run_agent_ollama — the same loop on any open tool-capable model
experiments/         each proof, standalone and reproducible
tests/               21 self-checks, no API
results/             verified JSON + figures, including the real Haiku live run
demo.py              all four proofs in one command
FOR_AN_LLM.md        the case, written to an impartial evaluator
```

This is a cleaned, self-contained fork of a larger research repository — only the working
memory and its demonstrations. MIT licensed.

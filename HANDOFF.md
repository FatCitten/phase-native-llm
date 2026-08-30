# HANDOFF — read this first

**Purpose of this file:** let a *fresh* Claude Code session (or any harness) pick up this repo
exactly where the last one left off, with no prior context. It offloads the whole project arc,
every committed result with its honest numbers, the code map with exact reusable signatures, the
environment constraints, the honesty ethos, the git state, and — most importantly — the **complete
plan for the next unbuilt piece (Extension VI)**, copied verbatim because it lives *outside* the repo
in the session's private plan file and would otherwise be lost.

**Last updated:** 2026-08-30 · **Branch:** `claude/custom-llm-training-arch-38028m` · **HEAD:**
`1066260` ("A society of spines…") · **Working tree:** clean.

---

## 0. TL;DR for the next session

- This repo proves, with **honest numpy numbers on CPU** (no GPU, no ML framework, no network), that
  **biological-style structure/consolidation** — overproduce → prune → **freeze (append-only)** →
  grow further out — gives a network things **traditional structure-blind training cannot**: axioms
  that stay **byte-identical**, **zero-forgetting** task addition, an **interpretable least-path-of-
  resistance trace** for every output, and (with breadth) **better capability-per-synapse**.
- **8 experiments are DONE, committed, tested, and documented in `README.md`** with their honest wins
  *and honest negatives*. See §3.
- **The next task is Extension VI** — a **promptable proof-of-concept** (tiny char-level LM + a glyph
  image classifier) on *real* offline data, packaged as a **Python CLI (numbers)** AND an
  **interactive web artifact**. It is **fully planned but NOT started**. The plan is in §6, verbatim.
- **The user's north star:** *"Be cold and impartial — we can only work with what 'is'."* Never dress
  up a weak result as a strong one. Report negatives as plainly as positives. Every experiment states
  its **kill-criteria up front** and honors them.

**To resume immediately:** read §6 (Extension VI plan), then start Deliverable A — create the `demo/`
package. First concrete step is spelled out in §8.

---

## 1. The project arc (how we got here)

This repo has three lineages, oldest to newest. **Only the third is the current frontier**; the first
two are finished and still present (don't delete them).

1. **Zero-parameter phase solver** (earliest). A `ZkBundleExplicit` net solves modular addition at
   **100% test acc with 0 learnable params, 0 gradient steps** — because the Fourier geometry *is* the
   solution. Thesis: "grokking is the cost of discovering structure, not a phase transition." Lives in
   `experiments/zkbundle_explicit_v2c.py`, `zero_param_demo.py`, `grokking_*`, and `PARADOX.md`.
2. **LLM-driven phase memory** (`phase_native/`, `MEMORY-AGENT.md`). An LLM drives an **O(1)** phase-
   native associative memory: solidify reasoning into low-byte nuggets, recall in constant time,
   compose multi-hop. Live-verified once on the API (Haiku 4.5, 8/8). **Note:** `--live` paths need an
   Anthropic key and network — **neither exists in this sandbox** — so those are not re-runnable here;
   everything else is.
3. **Consolidation arc (CURRENT).** The biological developmental loop as a growing/pruning numpy net,
   extended over 7 experiments, now heading into Extension VI (the promptable PoC). **This is where
   all current work happens.** §3 is the map.

The narrative thread the user cares about (in their words, paraphrased across the session): a network
should **overproduce then prune** like a brain; prior decisions become **inviolable axioms**; new
structure grows **further from the axiom** (distance = the amount of *non-relation*/void to the frozen
base); concepts **tighten into cross-paths** (a sparse mesh, not parallel columns); many teachers share
one **frozen spine** with per-task **branches** (no forgetting = "conquering confusion"); the spine
**grows and grafts** ("small brain + small brain = big brain", the graft preserving each source *by the
inviolability of its structure*); the world itself becomes a teacher; and a **society of spines** cross-
teaches to resist generational collapse. Extension VI finally exposes all this to **real data** and makes
it **promptable**.

---

## 2. Environment constraints (verified this session — do not re-learn the hard way)

- **Python + numpy + matplotlib** are present. **`sklearn` is NOT installed. No network. No GPU. No
  torch/tensorflow/jax** for the current arc (numpy only, manual backprop). CPU only.
- Because there's no network and no sklearn, **"real data" for Extension VI means committed public-
  domain text + matplotlib-rendered glyph bitmaps** — both offline and reproducible. Do NOT plan to
  download MNIST or fetch a corpus.
- Some experiments take **minutes** to run. Prefer launching long runs in the **background** and
  re-plotting figures from the saved JSON rather than re-running a 10-minute job just to tweak a plot.
- `matplotlib.use("Agg")` before importing pyplot (headless). Every experiment already does this.
- The Anthropic/Ollama API keys pasted earlier in the session are **ROTATED/dead and were never
  committed** — keys live only in the session scratchpad, never in the repo. Keep it that way.

---

## 3. What is DONE — the 8 committed experiments (with honest numbers)

All numbers below are the honestly-reported results already written into `README.md`. Each experiment
saves `results/<name>.{json,png}` and has tests in `tests/test_phase_native.py`. Kill-criteria were
stated up front and honored; **negatives are reported, not hidden**.

1. **`experiments/synaptic_pruning.py`** — *blindness + amplification.* A trained MLP over-produces
   synapses: **~50% removable at ~unchanged test acc**; capability-per-surviving-synapse rises up to
   **~40×**. On modular addition the net hits **100% train / 0% test** — objective satisfied, nothing
   understood (loss sees only the landing spot). Winning-ticket (init+mask) relearns faster than a
   random subnet of equal sparsity (Lottery Ticket).
2. **`experiments/consolidation_rounds.py`** — *the iterative loop (core substrate).* overproduce →
   train (frozen base untouched) → relational-prune the void → freeze append-only.
   - **Axioms inviolate** — round-1 survivors **byte-identical** after every later round.
   - **Void is real** — 3–12% of candidate units and ~40% of candidate connections relate to nothing,
     pruned every round at no accuracy cost.
   - **Tightening ratio flips efficiency** — loose loop reached comparable acc only with *more* wires
     (0.703 @ **7,238** vs dense 0.733 @ 2,976). Adding τ (ramped **0→0.8**, force+reward+pressure)
     pulling each new fiber onto **≤6 existing fibers** — a sparse **cross-path mesh** (564 fiber→fiber
     edges vs 4,879 loose) — cuts wiring **3.3×** (7,238→**2,210**) at ~equal acc (0.708), so
     **cap/synapse 0.32e-3 beats dense 0.25e-3**. τ ceiling 0.8 chosen by a transparent sweep
     (`consolidation_tau_sweep.py`).
   - **Honest limit** — on a *single* task, tightening trades depth for cheap wires (mean distance
     1.51 < loose 1.83), and **plain magnitude-pruning stays most wire-efficient (0.735 @ 2,210)**.
     Consolidation's single-task return is inviolable/legible structure, *not* beating pruning. The
     efficiency win needs **breadth** → next experiment.
3. **`experiments/multi_teacher.py`** — *many teachers, one frozen spine + branches.* Inference is a
   **least-path-of-resistance** walk (the figure traces one).
   - **Reuse** — 4 teachers @ **mean 0.760** using **1,875** wires vs a monolithic multi-task net
     pruned to the *same* budget @ **0.685** (**+7.5 pts** — shared primitives amortize).
   - **No forgetting** — tasks added one at a time: task-0 acc preserved **exactly (0.761→0.761→
     0.761→0.761)** while a fine-tuned monolith **collapses 0.84→0.24**.
4. **`experiments/spine_growth.py`** — *growing / self-specializing / graftable brains* (two wins, one
   honest negative).
   - **Growth** — balance-gated spine grows; later teachers reuse it more (**reuse fraction 0→0.57**)
     at held acc. **But efficiency does NOT beat the no-promotion baseline** — forced reuse causes
     *negative transfer* (a branch under-builds on a partial spine). **Reported, not hidden.**
   - **Shortcut node** — a distance-1 node distills a hot distance-2 fiber (**r²=0.87**), halving the
     path resistance at preserved acc.
   - **Grafting (small+small=big)** — two brains on disjoint input halves graft; a cross-task needing
     **both** is solved @ **0.91** while either brain alone reaches ~0.60 and from-scratch 0.31 — and
     **each source brain is preserved byte-for-byte**.
5. **`experiments/world_teacher.py`** — *the invariant ground resists generational collapse.*
   Rebuilding from scratch each generation drifts **0.742→0.635** over 8 gens; the axiom-grounded
   structure holds **0.742→0.690** (half the loss). Phase **magnetism toward anchors** (mean axiom-
   pointer directions) keeps alignment **0.85 vs 0.77**. Honest limit: on *accuracy* the structural
   ground does the work; the magnet adds geometric consistency, not capability, on this toy.
6. **`experiments/society.py`** — *a society of spines* (the current HEAD commit).
   - **Lone generative self-teaching collapses** — 0.692→0.605 over 6 gens (trains on its own outputs).
   - **The society collapses less** — cross-teaching (each spine trains on its *peers'* opinionated
     data, never its own) holds it above the lone spine at **every** generation (mean **0.642 vs
     0.621**), with **independence preserved** (pairwise disagreement ~0.10, not homogenized).
     **Honest limits:** the rescue is *modest* (self-generated data can't beat the ensemble's own
     accuracy — a ceiling), and **flaw-break-reform adds nothing** over cross-teaching here. Mechanism
     holds; magnitude bounded by the toy.

(The `consolidation_tau_sweep.py` supports #2's τ-ceiling choice; `beat_the_file.py`, `lucidity.py`,
`compose_multihop.py`, `seek_scaling.py`, `memory_agent_*` belong to lineages 1–2.)

---

## 4. Code map — the reusable substrate (exact signatures)

Everything Extension VI needs already exists. **Reuse, don't reinvent.** Exact signatures as committed:

**`experiments/consolidation_rounds.py`** — the core net.
```python
def softmax(z): ...
def fiber_distance(w_abs, src_dist): ...           # relational distance-from-axiom
def hier_teacher_data(D=20, hid=(10,10), C=4, N=4000, ntr=3000, seed=0): ...

class ConsolidatingNet:
    def __init__(self, D, C, seed=0): ...
    # frozen state (append-only, inviolate):
    #   self.frozen_W : list[np.ndarray]  kept incoming weights per round
    #   self.frozen_V : list[np.ndarray]  kept readout weights per round (for LPR traces)
    #   self.frozen_b : list[np.ndarray]  kept per-fiber biases per round (forward on NEW inputs)
    #   self.dist     : list[float]       distance-from-axiom of each established fiber
    #   self.bias     : np.ndarray (C,)   readout bias
    #   self.synapses, self.cross_edges : int
    @staticmethod
    def _cap_cols(M, k): ...                        # keep top-k columns by magnitude (sparse caps)
    def acc(self, y, which="te"): ...
    def seed_base(self, Ftr, Fte, dist, frozen_W=None): ...   # preload a shared frozen SPINE
    def grow_round(self, Xtr, ytr, Xte, yte, P=32, epochs=1500, lr=0.05, wd=1e-4,
                   floor=0.1, conn_floor=0.2, refit=400, tau=0.0, k_par=6,
                   prune_density=None, anchors=None, magnet=0.0): ...   # ONE consolidation wave
def monolithic(Xtr, ytr, Xte, yte, D, C, H, epochs, target_syn=None, lr=0.05, wd=1e-4, seed=1): ...
def run_loop(Xtr, ytr, Xte, yte, D, C, taus, P=32, epochs=1500, seed=1): ...
```

**`experiments/society.py`** — forward a frozen net on ARBITRARY new inputs (this is how you make the
consolidation net *promptable* / generative). **Critical for Extension VI.**
```python
def forward_logits(net, X): ...    # recompute every frozen fiber on X, then the readout -> logits
def predict(net, X): ...           # argmax of forward_logits
def acc_on(net, X, y): ...
def make_spine(Xtr, ytr, Xte, yte, D, C, seed, rounds=2, P=32, EP=800): ...
def forward_feats(net, X): ...     # per-round fiber activations (for attribution / refit)
def flaw_break_reform(net, Xg, target, harm_frac=0.15, refit=300, lr=0.05, wd=1e-4): ...
```

**`experiments/multi_teacher.py`** — spine/branch pattern + the interpretable trace.
```python
def shared_primitive_teachers(T=4, D=12, prim=6, C=4, N=4000, ntr=3000, seed=0): ...  # the TRUE world
def build_spine(Xtr, y0tr, Xte, y0te, D, C, P=32, EP=1200, seed=1): ...
def grow_branch(spine, Xtr, ytr, Xte, yte, D, C, P=32, EP=1200, seed=2): ...
def spine_incoming(net, D): ...                    # which inputs/fibers feed each fiber
def trace_lpr(branch, spine, x_i, i, D): ...       # least-path-of-resistance source->destination trace
def layout(spine, branches, D, T): ...             # radius=distance, angle=branch (spine/spiral figure)
```

**`experiments/spine_growth.py`** — growth gates, shortcut distillation, grafting.
```python
def growing_teachers(...); def gate_shared(...); def gate_balanced(...); def is_novel(...)
def grow_sequentially(...); def fit_relu_node(...); def distill_shortcut(...)
def two_brain_teachers(...); def build_bank(...); def graft_brains(...); def solo_acc(...)
```

**`experiments/synaptic_pruning.py`** — the traditional baseline to beat/tie.
```python
class MLP:
    def __init__(self, d_in, d_hidden, d_out, seed=0); def forward(self,X); def step(self,X,y,lr,wd)
    def acc(self,X,y); def n_active(self); def n_weights(self)
def train(net, Xtr, ytr, Xte, yte, epochs, lr, wd, log_every=0): ...
def magnitude_mask(net, keep_abs): ...
def teacher_data(D=20, Ht=8, C=4, N=4000, ntr=3000, seed=0): ...
def modular_data(k=12, train_frac=0.75, seed=0): ...
```

**`tests/test_phase_native.py`** — 13 test fns run by `main()`; current arc ones are
`test_consolidation, test_multi_teacher, test_spine_growth, test_world_teacher, test_society`.
Run: `python tests/test_phase_native.py` → prints `*** ALL TESTS PASSED ***` or exits 1.

---

## 5. How to run / verify

```bash
# the full current arc (each saves results/<name>.{json,png}); minutes each, launch in background
python experiments/synaptic_pruning.py && python experiments/consolidation_rounds.py \
  && python experiments/multi_teacher.py && python experiments/spine_growth.py \
  && python experiments/world_teacher.py && python experiments/society.py

# all tests
python tests/test_phase_native.py
```

---

## 6. NEXT TASK — Extension VI (fully planned, NOT started). Verbatim plan.

This is the active, approved, unbuilt work. The user asked (this session) for a **promptable LLM proof
of concept** proving the structure's validity/utility on datasets vs traditional LLM training, and via
`AskUserQuestion` chose **PoC type = Both** (rigorous classifier validity **and** a promptable text
generator) on **Surface = Both** (a Python CLI with numbers **and** an interactive web artifact). The
plan below is copied verbatim from the session's external plan file so nothing is lost.

> ### Extension VI — The proof of concept: a promptable model trained our way, on real data
>
> **Context (why).** Seven experiments proved the consolidation method's structural advantages on
> synthetic teachers. Now expose the structure to REAL data and package a **promptable proof-of-
> concept** demonstrating utility vs traditional LLM training — as a Python CLI (numbers) AND an
> interactive web artifact. **Honest scope up front:** this is a TINY, char-level, fixed-context model
> plus a small image classifier — NOT a large language model, NOT fluent. The claim is about the
> *training method*, not scale: a model trained our way can do what traditional training cannot — **add
> knowledge with zero forgetting** and **explain every output by tracing its structure** — shown with
> numbers on real data and a promptable interface. We never dress up weak generation as competence.
>
> Env: numpy + matplotlib present, **no sklearn / no network** — so "real data" is a committed public-
> domain text corpus + rendered glyph images, both offline and reproducible.
>
> **Deliverable A — a promptable char-level model (the "LLM" PoC, the star).**
> - Real text: a small committed public-domain corpus (`demo/corpus.txt`, a few KB). Char vocab V.
> - Model: `demo/charlm.py` wraps `ConsolidatingNet` as a fixed-context next-char predictor (window W of
>   one-hot chars → `[W×V]` input, C=V output). Reuse `grow_round` (spine + tightening + pruning),
>   `society.forward_logits` (forward on NEW contexts), an autoregressive `generate(prompt, temp)`, and
>   `multi_teacher.trace_lpr`/`spine_incoming` for a per-token **source→destination** trace.
> - Rigorous validity: held-out **next-char accuracy / bits-per-char** vs a matched traditional MLP
>   (`synaptic_pruning.MLP`+`train`). Reported honestly (it learns local structure, not fluency).
> - Structural advantages (the real point), each a measured number vs traditional:
>   - **No forgetting** — train on corpus A, add corpus B as a frozen BRANCH (reuse the multi_teacher /
>     spine_growth branch pattern) → A's next-char accuracy preserved EXACTLY; a fine-tuned MLP forgets A.
>   - **Interpretable generation** — every generated char carries its least-path-of-resistance trace.
>   - **Efficiency** — capability-per-synapse vs the traditional MLP.
> - Promptable: `python demo/demo.py --prompt "..."` → continuation + trace.
>
> **Deliverable B — real-data validity on a recognizable dataset (breadth across modality).**
> - Real images offline: `demo/glyphs.py` renders the 10 digit glyphs to small bitmaps via matplotlib
>   with jitter + noise → a recognizable image-classification dataset (described accurately as *rendered
>   glyphs*, not MNIST). Deterministic fallback to a procedural glyph set if rendering is flaky.
> - The consolidation classifier vs a traditional MLP: accuracy + the SAME structural advantages — **no
>   forgetting** (train digits 0–7, add 8–9 as frozen branches → 0–7 preserved vs traditional forgets),
>   **interpretable** prediction trace, **efficiency**. Shows the method carries across modalities.
>
> **Deliverable C — the surfaces: CLI (numbers) + web artifact (interactive).**
> - CLI `demo/demo.py`: trains A + B, prints a head-to-head **comparison table** (accuracy; retention
>   after adding new classes/domain, ours vs traditional; cap/synapse); generates from prompts with
>   traces; saves `results/demo.{json,png}`; and **exports** the char-LM to `results/charlm.json` (rounds
>   of frozen_W/V/b, vocab, window) for the web.
> - Web artifact (built with the Artifact tool; **load `artifact-design` first**): a single honest HTML
>   page with the exported model **inlined** as a JS const (a tiny net → tens of KB), a JS
>   reimplementation of `forward_logits` + autoregressive sampling + the trace, and a prompt box —
>   interactively promptable and shareable. Framed explicitly as a **training-method proof of concept (a
>   tiny char model, not a fluent LLM)**; shows live generation + the structural trace, plus the no-
>   forgetting / efficiency numbers as static panels. Must NOT impersonate a real product or LLM.
>
> **Files & reuse.** New `demo/` package: `corpus.txt`, `charlm.py`, `glyphs.py`, `demo.py`. Reuse
> `ConsolidatingNet` (`grow_round`, `seed_base`, `frozen_W/V/b`, `acc`), `society.{forward_logits,
> predict}`, `multi_teacher.{trace_lpr, spine_incoming}`, `synaptic_pruning.{MLP, train, magnitude_mask}`.
>
> **Tests** (`tests/test_phase_native.py`): char-LM windowing + deterministic generation; the JSON
> export round-trips the forward pass (numpy vs the exported-shape recompute, so the JS port is
> trustworthy); no-forgetting on the glyph classifier (old classes byte-identical after adding new); the
> traditional baseline forgets.
>
> **Verification.**
> 1. `python demo/demo.py` → prints validity + the structural-advantage comparison for text and glyphs;
>    generates sample continuations with traces; saves `results/demo.{json,png}` + `results/charlm.json`.
> 2. `python tests/test_phase_native.py` → all pass incl. the new demo checks.
> 3. Publish the web artifact and hand back the URL.
> 4. Commit to `claude/custom-llm-training-arch-38028m`.
>
> **Honest kill-criteria / guardrails (up front).**
> - The char-LM is TINY and NOT fluent: report next-char accuracy and show honest (imperfect)
>   generations — never present gibberish as competence. The web page states plainly it is a training-
>   method PoC.
> - If the consolidation model does NOT beat the matched traditional MLP on raw accuracy, say so — the
>   win is the STRUCTURAL advantages (no-forgetting, interpretability), which hold by construction;
>   report parity.
> - No network dependence: real data is a committed public-domain text + rendered glyphs. If glyph
>   rendering is flaky, use the deterministic procedural glyph set and say we did.

**Design note for the char-LM (learned this session):** the reason `frozen_b` (per-fiber bias) was added
to `ConsolidatingNet` is precisely so `society.forward_logits` can evaluate a frozen net on **new,
unseen inputs** (new char contexts / new prompts). That is the mechanism that makes the consolidation net
generative/promptable — build on it, don't rebuild it.

---

## 7. Honesty ethos + git rules (non-negotiable, carry forward)

- **Impartiality.** *"Be cold and impartial — we can only work with what 'is'."* State kill-criteria up
  front; report negatives (efficiency-vs-pruning on single tasks, negative transfer in spine growth,
  break-reform neutrality, the society's bounded ceiling) as plainly as wins. Do not force a result.
- **Never commit secrets.** No API keys in the repo, ever. Scratchpad only.
- **Branch discipline.** Develop, commit, and push ONLY to `claude/custom-llm-training-arch-38028m`.
  Create it from the default branch if missing. Do NOT push elsewhere without explicit permission.
- **Commit footer** (exactly, every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LJp3S2qYagi8uBwuL81636
  ```
  **No model identifiers anywhere else** in commits/PRs/code/artifacts — chat replies only.
- **Push:** `git push -u origin claude/custom-llm-training-arch-38028m`; on *network* failure only,
  retry up to 4× with backoff (2s, 4s, 8s, 16s).
- **No PR unless the user explicitly asks.** If asked, check for a PR template first.
- If the branch's PR was already merged, treat follow-up as a fresh change: restart the branch from the
  latest default branch (same name) and push there — never stack on merged history.

---

## 8. Exact next step for the resuming session

1. Create the `demo/` package. Start with **`demo/corpus.txt`** (a few KB of committed public-domain
   text — e.g. a passage from a pre-1900 work) and **`demo/charlm.py`**:
   - vocab + `window(text, W)` → `(X one-hot [N, W*V], y next-char [N])`;
   - `train_charlm(...)` using `ConsolidatingNet.grow_round` (a couple of rounds, tau ramp, `prune_density`);
   - `generate(net, prompt, n, temp)` using `society.forward_logits` autoregressively;
   - `bits_per_char` / next-char accuracy on a held-out tail;
   - `add_branch(...)` for corpus B (the no-forgetting demo) and `export_json(net, ...)` for the web.
2. Then `demo/glyphs.py` (Deliverable B), then `demo/demo.py` (Deliverable C CLI + `results/charlm.json`).
3. Extend `tests/test_phase_native.py` with the four Extension-VI checks (§6 Tests), wire them into
   `main()`, run `python tests/test_phase_native.py`.
4. Build the **web artifact** (load the `artifact-design` skill first; inline `results/charlm.json`;
   reimplement `forward_logits` + sampling + trace in JS; honest framing; never impersonate a real LLM).
5. Update `README.md` with the Extension-VI section (honest numbers), commit to the branch, hand back
   the artifact URL.

**Sanity check before you start:** `git status` should be clean on `claude/custom-llm-training-arch-38028m`
at HEAD `1066260`; `python tests/test_phase_native.py` should print `*** ALL TESTS PASSED ***`.

---

*Earlier lineage handoffs (kept for provenance, not current): `MEMORY-AGENT.md` (phase memory),
`PARADOX.md` (grokking redefinition), `minimal-handoff.md`, `IMPLEMENTATION-ROADMAP.txt`,
`PHASE-NATIVE-CONTEXT.txt`. The current story is §1–§8 above and the "synaptic pruning → consolidation"
section of `README.md`.*

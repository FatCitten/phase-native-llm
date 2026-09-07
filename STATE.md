# NAUTILUS — STATE.md (single source of truth)

> Read on session entry. Update on session exit. This is the PhD coworker's working
> memory. Trust this + git log, NOT the stale README.

## The Goal (honest framing — the harness enforces this)

Train a small Nautilus model that beats a matched traditional MLP on next-word
accuracy, using distillation from a frontier LLM as the teacher, and prove the
structural wins (no-forgetting, capability-per-synapse, legibility) that a plain
MLP cannot match.

NOT a frontier-scale LLM. Laptop hardware (Intel Arc iGPU, 15GB RAM, CPU numpy,
no CUDA). Any experiment drifting toward "train a big model" gets flagged.

## Verified State (as of 2026-09-07)

- Frontier branch `claude/custom-llm-training-arch-38028m` checked out on disk at
  `/home/meowar/areeyh/phase-native-llm`. Master is STALE.
- venv `.venv` set up (numpy, scipy, matplotlib, anthropic).
- Test suite: **1 FAILURE** — `recall mask prunes some fibers`.
  - Root cause: config artifact. With 2 rounds / 50 epochs / tiny corpus, every
    fiber fires on the test set, so the mask is valid (masked==full passes) but
    not selective (total == all_fibers). NOT a correctness bug.
  - Fix direction: use a config where pruning actually happens (more rounds /
    more fibers / sparser data), don't weaken the assert.
- Suite takes ~8 min — VIOLATES FAST OR USELESS. Needs shrinking to <30s.
- Word-level next-token: Round 1 hit 24.97% next-word acc on 457-word vocab
  (~100x chance). Real language signal.

## The 4 Pillars (what Nautilus IS)

1. **Legible** — StructureEngine / NautilusVisualizer: observe every fiber's
   distance-from-axiom, sources, readout. One source of truth, two viewers.
2. **Editable** — edit fibers (zero/prune/rewire/add, set readout) with append-only
   safety guards. An LLM (llm_play.py) explores and edits live.
3. **Durable** — save_structure / load_structure / read / write. Round-trips exactly.
4. **Collaborative** — live website where human + LLM watch training and collaborate.

## Honest Results (with numbers)

| Claim | Result | Status |
|-------|--------|--------|
| Word next-token (Round 1) | 24.97% on 457-word vocab | verified |
| Char next-char (Extension VI) | 0.190 vs MLP 0.245 | MLP wins raw acc (expected) |
| Glyph digit acc | 0.877 vs MLP 0.938 | MLP wins raw acc |
| Glyph cap/synapse (x1e-3) | 0.43 vs MLP 0.01 | Nautilus ~43x |
| Glyph no-forgetting (0-7 after 8-9) | 0.877 preserved vs MLP 0.923->0.000 | Nautilus wins |
| Text no-forgetting | corpus A/B same task, nothing to forget | honest negative |

## Open Threads / Next Experiments (pick one)

1. **Matched-MLP comparison on the dashboard** — side-by-side Nautilus vs MLP on
   next-word acc. The honest "superiority" proof. (Recommended.)
2. **Fix the failing test** — config where recall mask actually prunes.
3. **Shrink the test suite** — get it under 30s (FAST OR USELESS).
4. **Run foster.py distillation** — frontier LLM as teacher, child trains on soft
   targets. The proven path to a competitive small model.
5. **Expand corpus** — more public-domain works for room to grow.

## Kill-Criteria (state up front before any experiment)

- (To be filled per experiment. Example: "If Nautilus does not beat the matched
  MLP on next-word acc after distillation, the distillation path is not working
  and we pivot to structural wins only.")

## Git Discipline

- Work ONLY on `claude/custom-llm-training-arch-38028m`.
- Push after every meaningful commit (this is the only copy on disk).
- Do not push to master without explicit permission.
- Never commit secrets.

## Key Facts

- User: Aaron. Rack-and-stack data center tech, self-taught. Wants to make games
  and "cool apps." Arch + sway, TUI/micro, GitHub heavy.
- The whole Nautilus idea is Aaron's: "if the architecture exists as bytes and can
  be edited, it can be saved — so it's a machine that stores/loads/reads/writes,
  dynamically engineered by LLMs."
- Honesty ethos (north star, non-negotiable): "Be cold and impartial — we can only
  work with what 'is'." Report negatives as plainly as positives. State kill-criteria
  up front.

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
- Test suite: **ALL PASS**, runs in **~10s** (was 8 min — the `_tiny_word_data()`
  refactor fixed FAST OR USELESS). Command: `python tests/test_phase_native.py`.
- The remote has moved past the old handoff: current frontier is the
  **signal-engine / guide-loop** arc (see below).

## The Current Frontier — "the LLM guides, the structure decides"

The headline loop (`demo/guide_loop.py`): a frontier LLM emits TWINGES (soft
next-word distributions on contexts). The child grows a consolidation round with
soft targets — its OWN overproduce->prune->freeze dynamics decide which signals
STICK. The teacher GUIDES, the child's structure DECIDES.

Key files (all under `demo/`):
- `signal_engine.py` — SignalEngine: teacher_twinge, grow_on_signals,
  refine_on_signals (nudge readout, no new round), digest_round (prune
  non-load-bearing scaffolding), trauma_collapse (collapse tipping-point fibers,
  rebuild from survivors), _recompute_base.
- `guide_loop.py` — the headline loop. Run:
  `python -m demo.guide_loop --model glm-5.3-flash --rounds 3 --contexts 20`
  (default mode=refine; grow mode adds a new round).
- `metrics.py` — capability_per_synapse, no_forgetting, phi_diagnostic (does the
  structure's round-count ratios approach the golden ratio 1.618 as it collapses?
  DIAGNOSTIC — phi should EMERGE, not be imposed).
- `foster.py` — distillation harness (soft targets from frontier LLM).
- `superiority.py` — honest MLP-vs-consolidation comparison.
- `instrument.py` + `instrument.html` — graphical load/trace/edit tool.
- `engine.py` / `visualizer.py` — StructureEngine / NautilusVisualizer.
- `llm_play.py` — tool-calling loop: LLM engineers the structure live.
- `wordlm.py` — word tokenizer, vocab, windowing, next-word training.
- `PLUGIN_CONTRACT.md` — the one shape any Nautilus machine carries.

## Honest Results (with numbers)

| Claim | Result | Status |
|-------|--------|--------|
| Word next-token (Round 1) | 24.97% on 457-word vocab | verified |
| Char next-char (Extension VI) | 0.190 vs MLP 0.245 | MLP wins raw acc (expected) |
| Glyph digit acc | 0.877 vs MLP 0.938 | MLP wins raw acc |
| Glyph cap/synapse (x1e-3) | 0.43 vs MLP 0.01 | Nautilus ~43x |
| Glyph no-forgetting (0-7 after 8-9) | 0.877 preserved vs MLP 0.923->0.000 | Nautilus wins |
| Text no-forgetting | corpus A/B same task, nothing to forget | honest negative |
| recall_mask pruning | every surviving fiber fires somewhere; mask is a no-op on trained net | finding |

## Open Threads / Next Experiments (pick one)

1. **Run guide_loop with a real teacher** — the headline experiment. Needs the
   OLLAMA key. Measure cps + no-forgetting + phi after each round. Honest framing:
   if twinges don't improve metrics, report it plainly.
2. **Matched-MLP comparison on the dashboard** — side-by-side Nautilus vs MLP on
   next-word acc. The honest "superiority" proof.
3. **Expand corpus** — more public-domain works for room to grow.
4. **phi_diagnostic** — does phi actually emerge from repeated collapse? Currently
   a diagnostic; needs a real run to see if the ratios approach 1.618.

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

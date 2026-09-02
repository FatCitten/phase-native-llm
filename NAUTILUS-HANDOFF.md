# NAUTILUS — SESSION HANDOFF (start a fresh session here)

**Repo:** `github.com/FatCitten/phase-native-llm`, branch `claude/custom-llm-training-arch-38028m`
**Local clone:** `/tmp/phase-native-llm` (VOLATILE — /tmp is wiped on reboot. Re-clone if gone.)
**Latest commit as of handoff:** `1b7ca97` (fully pushed, clean tree)

---

## 1. THE PROJECT (what we're building)

**Nautilus** = a consolidation neural network (grows structure in waves, prunes void, freezes
axioms) whose internal structure is **legible, editable, and durable**. It IS a machine: it
can be **stored / loaded / read / written**, and an LLM can engineer it via tool calls.

The 4 pillars:
1. **Legible** — StructureEngine / NautilusVisualizer let you observe every fiber's
   distance-from-axiom, sources, readout. One source of truth (`NautilusVisualizer.view()`),
   two viewers: `to_llm()` (what the LLM reads) and `to_human()` (HTML).
2. **Editable** — edit fibers (zero/prune/rewire/add, set readout) with append-only safety
   guards. An LLM (`llm_play.py`) explores the structure and edits it live.
3. **Durable** — `save_structure` / `load_structure` / `read` / `write`: the structure is
   bytes on disk. Round-trips exactly (tests verify ~1e-9).
4. **Collaborative** — a live website where a human and an LLM watch training and collaborate.

## 2. THE CURRENT FRONTIER — live next-token training website

We retargeted from char-level to **WORD-LEVEL NEXT-TOKEN prediction** on a ~15k-word
public-domain corpus (Alice, Pride & Prejudice, Moby Dick). Round 1 hit **24.97%** next-word
accuracy on a 457-word vocab (~100x chance) — real language signal.

### The live system (3 processes)
A FastAPI server runs a background consolidation training loop, streams live events to a
browser over SSE, and exposes a command API. A separate agent driver (an LLM) watches and
collaborates with the human in a chat pane.

**Files (all under `demo/`):**
- `train_server.py` — FastAPI + SSE backend; trains the model in a thread; `/events` (SSE),
  `/command` (start/pause/resume/stop, chat, agent_reply, directive, get_state)
- `dashboard.html` — frontend: live metrics, acc-over-rounds chart, structure-being-established
  (fiber bars scaled by distance), agent directive, human<->agent chat
- `agent_driver.py` — the "me" side: polls the server for new human messages + state, and
  replies via an LLM (Ollama Cloud, `glm-5.3-flash`); can issue training commands
- `wordlm.py` — word tokenizer, vocab, windowing, next-word training + generation
- `web_nautilus.py` + `results/nautilus_demo.html` — a self-contained interactive demo
  (observe/trace/edit/save-load), model-agnostic plugin contract
- `visualizer.py` — `NautilusVisualizer` (view / to_llm / to_human / trace)
- `engine.py` — `StructureEngine` (observe/trace/edit/persist)
- `PLUGIN_CONTRACT.md` — the one shape any Nautilus machine must carry:
  `{ window, vocab, bias, dist, rounds }`; documents invariants + round-trip guarantee

### Restart commands (from the repo, `.venv` activated)
```bash
cd /tmp/phase-native-llm && source .venv/bin/activate
# 1) training server  → dashboard at http://localhost:8000
python -m demo.train_server
# 2) agent driver (the LLM collaborator) — needs the OLLAMA key from ~/.hermes/.env
export OLLAMA_API_KEY=$(grep -E '^OLLAMA_API_KEY=' ~/.hermes/.env | head -1 | cut -d= -f2-)
python -m demo.agent_driver --model glm-5.3-flash --interval 3
```
On the dashboard press **Start**. The agent auto-posts directives and replies to human chat.

### Running processes at handoff (may have died if this session ended)
- `train_server` (was pid ~290818) — server
- `agent_driver` (was pid ~253252) — agent
The training loop does NOT persist across server restarts (state is in-memory). A fresh
`start` rebuilds from corpus round 0. The last live snapshot was ~round 8, acc ~0.25.

### Known honest limitations (do NOT oversell)
- **Tiny model, small corpus** — will NOT match frontier next-token accuracy (those are
  100B+ params / trillions of tokens). The honest "superiority" target is **beating a matched
  traditional MLP** on next-word accuracy, plus the structural wins.
- Word-level training is SLOW per round (D=1828, C=457, ~16k samples, 400 epochs → minutes/round).
- The agent driver is tied to whichever session launched it; it doesn't survive /tmp wipes.

## 3. TESTS — ALL PASS
```bash
python tests/test_phase_native.py   # -> *** ALL TESTS PASSED ***
```
Covers: consolidation, multi-teacher, spine-growth, world-teacher, society, **Nautilus
persistence (store/load/read/write round-trip)**, **visualizer (LLM+human views agree)**,
edit safety guards (prune referenced fiber refused, add_fiber-to-earlier-round refused).

## 4. OPEN THREADS / NEXT STEPS (pick one)
1. **Add a matched-MLP comparison to the dashboard** — side-by-side Nautilus vs traditional
   MLP on next-word accuracy. This is the honest "superiority" proof. (Recommended.)
2. **Expand the corpus** further (more public-domain works) to give the model room to grow.
3. **More consolidation rounds / bigger config** on the word model to push accuracy higher.
4. Model-agnostic visualizer is done; could wire the web demo to the live-trained machine.

## 5. GIT DISCIPLINE
- Work ONLY on branch `claude/custom-llm-training-arch-38028m`. Create from master if missing.
- Commit footer (exactly, every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LJp3S2qYagi8uBwuL81636
  ```
- **Push after every meaningful commit** — /tmp is wiped on reboot and this is the ONLY copy
  of the work on disk. `git push origin claude/custom-llm-training-arch-38028m` works (gh auth
  is set up as user FatCitten).
- Do not push to master without explicit permission.

## 6. KEY FACTS
- User: **Aaron**, a data-center rack-and-stack tech, self-taught cod+some CS theory. Wants to
  make games and "cool apps." Runs Arch + sway, likes TUI/micro, uses GitHub heavily.
- The whole Nautilus idea is Aaron's: "if the architecture exists as bytes and can be edited,
  it can be saved — so it's a machine that stores/loads/reads/writes, dynamically engineered
  by LLMs." Every build here serves that idea.
- **Honesty ethos (Aaron's north star, non-negotiable):** *"Be cold and impartial — we can only
  work with what 'is'."* Report negatives as plainly as positives. Never dress up a weak result.
  State kill-criteria up front. This matters more than any metric.

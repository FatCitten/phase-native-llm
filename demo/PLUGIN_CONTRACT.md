# Nautilus Plugin Contract

The Nautilus machine is a **model-agnostic environment**: any structure — trained by a
person or engineered by an LLM — plugs into the same visualize / trace / edit / save-load
tooling. This document is the contract every producer and consumer must honor.

## The structure (one shape, any origin)

A Nautilus machine is a JSON object with exactly these fields:

```json
{
  "window": 8,
  "vocab": ["\n", " ", "a", "b", "c", "..."],
  "bias": [0.20, 1.36, -0.17, "..."],
  "dist": [1.0, 1.0, 1.03, "..."],
  "rounds": [
    { "W": [[0.02, 0.05, "..."], "..."], "V": [[0.1, "..."], "..."], "b": [0.01, "..." ] },
    { "W": [[...]], "V": [[...]], "b": [...] }
  ]
}
```

| field | type | meaning |
|-------|------|---------|
| `window` | int | context window W (chars) |
| `vocab` | array[str] | the character vocabulary, length V |
| `bias` | array[float] | readout bias, length C (= len(vocab)) |
| `dist` | array[float] | distance-from-axiom of every fiber, in round order |
| `rounds` | array | one entry per consolidation round |
| `rounds[i].W` | 2D array | incoming weights: `[D + width_before_i] x [n_fibers_i]` |
| `rounds[i].V` | 2D array | readout weights: `[n_fibers_i] x [C]` |
| `rounds[i].b` | array[float] | fiber biases, length `n_fibers_i` |

`D = window * len(vocab)`. `width_before_i = sum of n_fibers of rounds 0..i-1`.
`C = len(vocab)`.

## Producers (how a machine is created)

- **Python** — `engine.StructureEngine.save_structure(path)` writes exactly this shape
  (plus `D`/`C`/`synapses`). `engine.StructureEngine.read(path, D, C)` / `load_structure`
  rebuild a live engine from it.
- **LLM** — the `llm_play` harness edits a live structure via tools; the final structure is
  saved with `save_structure`. Any tool that emits a structure must emit this shape.
- **Browser** — the demo's "Save to file" writes this shape (with `window`+`vocab`).

## Consumers (how a machine is used)

- **Python** — `StructureEngine` (observe/trace/edit), `NautilusVisualizer` (LLM + human
  views), `llm_play` (LLM tool loop).
- **Browser** — `nautilus_demo.html` loads any structure with this shape via "Load from
  file" and re-derives all dimensions live (`curV`/`curW`/`curC`). It does NOT assume a
  fixed vocab/window/fan-out.

## Invariants (must hold for a valid machine)

1. `len(vocab) == len(bias) == C`.
2. `rounds[i].W` has `D + width_before_i` rows and `n_fibers_i` columns.
3. `rounds[i].V` has `n_fibers_i` rows and `C` columns.
4. `len(dist) == sum(n_fibers_i)`.
5. **Append-only structure**: a fiber in round `i` may be read by fibers in later rounds.
   Removing it (prune) is refused if a later round reads it. Adding a fiber is only allowed
   on the **last** round (adding to an earlier round would change its width, which later
   rounds read as input).

## Round-trip guarantee

`save_structure` → `read` reproduces the exact forward pass (verified to ~1e-9 in tests).
The browser's JS `forwardLogits` matches the numpy forward pass to ~1e-15. So a machine
engineered by an LLM in Python, saved, and loaded in the browser is the *same* machine.

## Validation

`tests/test_phase_native.py` covers: persistence round-trip, visualizer agreement
(LLM + human views), edit safety guards (prune referenced fiber refused, add_fiber to
earlier round refused), and the JS↔numpy forward-pass match.

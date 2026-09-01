"""
demo/web_nautilus.py — build an interactive, self-contained Nautilus demo (single HTML).

The demo is a browser app that lets you OBSERVE, TRACE, EDIT, and SAVE/LOAD a Nautilus
machine — a consolidation char-LM whose structure is legible and editable. Everything
(the model weights, the forward pass, the structure engine, the edit tools, save/load)
is inlined into one HTML file with no server, no network, no dependencies.

The model is a COMPACT Nautilus (small window/vocab/rounds) so the whole structure fits
inline. It is a tiny char-level next-char predictor on public-domain text — NOT a fluent
LLM. The demo is framed honestly as a training-method / machine proof of concept.

Run: python -m demo.web_nautilus  ->  writes results/nautilus_demo.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import charlm
from demo.demo import load_sections
from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits

DEMO = Path(__file__).resolve().parent
RESULTS = Path("results")


def train_compact():
    """Train a compact Nautilus char-LM on ALICE and return (net, vocab, W, Xte, yte)."""
    sections = load_sections(DEMO / "corpus.txt")
    corpus_a = sections.get("ALICE", "")
    vocab = charlm.build_vocab([corpus_a])
    W = 8
    Xa, ya = charlm.window(corpus_a, W, vocab)
    Xtr, ytr, Xte, yte = charlm.split_windows(Xa, ya, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)
    net = ConsolidatingNet(D, C, seed=1)
    for r in range(3):
        net.grow_round(Xtr, ytr, Xte, yte, P=24, epochs=300, tau=0.0,
                       floor=0.05, conn_floor=0.1)
    return net, vocab, W, Xte, yte


def export_compact(net, vocab, W):
    """Serialize the compact Nautilus for the JS engine (rounds + bias + dist)."""
    return {
        "window": W,
        "vocab": list(vocab),
        "bias": net.bias.tolist(),
        "dist": [float(d) for d in net.dist],
        "rounds": [{"W": w.tolist(), "V": v.tolist(), "b": b.tolist()}
                   for w, v, b in zip(net.frozen_W, net.frozen_V, net.frozen_b)],
    }


# The HTML template. JS uses template literals with {..} — so we build the page with
# placeholder substitution (__MODEL__, __ACC__, __W__, __V__) instead of f-strings to
# avoid brace-escaping hell.
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nautilus — an editable, durable machine</title>
<style>
  :root { --green:#2e7d32; --red:#c62828; --blue:#1565c0; --ink:#1a1a1a; --bg:#f6f7f9; --card:#fff; --line:#e2e4e8; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:20px; line-height:1.5; }
  .wrap { max-width:1000px; margin:0 auto; }
  h1 { font-size:1.6rem; margin:0 0 2px; }
  .sub { color:#555; font-size:.9rem; margin-bottom:14px; }
  .banner { background:#fff3cd; border:1px solid #ffe08a; border-radius:8px; padding:10px 14px; font-size:.82rem; margin-bottom:16px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:760px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
  .card h2 { font-size:1rem; margin:0 0 10px; }
  .card h3 { font-size:.85rem; margin:14px 0 6px; color:#333; }
  label { font-size:.8rem; font-weight:600; display:block; margin:8px 0 4px; }
  input[type=text], textarea { width:100%; padding:8px 10px; font-size:.95rem; border:1px solid #ccc; border-radius:7px; font-family:inherit; }
  button { background:var(--blue); color:#fff; border:none; padding:8px 14px; font-size:.85rem; border-radius:7px; cursor:pointer; margin:6px 6px 0 0; }
  button:hover { filter:brightness(1.08); }
  button.sec { background:#555; }
  button.green { background:var(--green); }
  button.red { background:var(--red); }
  .out { margin-top:10px; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9rem; background:#0f1115; color:#d7dae0; border-radius:8px; padding:12px; white-space:pre-wrap; word-break:break-word; min-height:60px; }
  .trace { margin-top:8px; font-size:.78rem; color:#8a8f98; font-family:ui-monospace,Menlo,Consolas,monospace; }
  .mono { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.8rem; }
  .stat { display:inline-block; background:#f0f1f4; border-radius:6px; padding:4px 10px; margin:2px 6px 2px 0; font-size:.8rem; }
  .fiber { border:1px solid var(--line); border-radius:7px; padding:8px; margin:6px 0; font-size:.78rem; }
  .fiber .h { font-weight:600; }
  .fiber .d { color:#666; }
  .fiber .src { color:#1565c0; }
  .fiber .ro { color:#2e7d32; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  .row input { width:auto; flex:1; }
  .foot { font-size:.75rem; color:#888; margin-top:20px; }
  .badge { display:inline-block; background:#e8f0fe; color:#1565c0; border-radius:4px; padding:1px 6px; font-size:.7rem; margin-left:6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Nautilus <span class="badge">editable · durable · LLM-engineered</span></h1>
  <div class="sub">A consolidation network whose structure you can observe, trace, edit, and save — a machine that lives as bytes on disk.</div>
  <div class="banner"><strong>Honest scope:</strong> this is a <em>tiny</em> char-level next-char model (window __W__, vocab __V__), <strong>not</strong> a fluent LLM. It learns local text structure. The point is the <em>machine</em>: legible, editable, durable structure. Baseline next-char accuracy: <strong>__ACC__</strong>.</div>

  <div class="grid">
    <div class="card">
      <h2>Generate</h2>
      <label for="prompt">Prompt (it predicts the next char, one at a time)</label>
      <input type="text" id="prompt" value="Alice was beginning to">
      <div class="row">
        <label style="margin:0">temp <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.8" style="width:90px;vertical-align:middle"></label>
        <button onclick="generate()">Generate</button>
        <button class="sec" onclick="clearOut()">Clear</button>
      </div>
      <div class="out" id="out">(generation appears here)</div>
      <div class="trace" id="gentrace"></div>
    </div>

    <div class="card">
      <h2>Observe the structure</h2>
      <div id="stats"></div>
      <h3>Growth log (structure being established)</h3>
      <div id="growth"></div>
      <h3>Fiber profiles</h3>
      <div class="row">
        <input type="number" id="fround" value="0" min="0" style="width:70px">
        <input type="number" id="ffiber" value="0" min="0" style="width:70px">
        <button onclick="showFiber()">Profile</button>
      </div>
      <div id="fiber"></div>
    </div>

    <div class="card">
      <h2>Trace a prediction</h2>
      <label for="tprompt">Input context</label>
      <input type="text" id="tprompt" value="Alice was beginning to">
      <button onclick="trace()">Trace</button>
      <div class="out" id="traceout"></div>
      <div class="trace" id="tracesteps"></div>
    </div>

    <div class="card">
      <h2>Edit the structure</h2>
      <div class="row">
        <input type="number" id="eround" value="0" min="0" style="width:70px">
        <input type="number" id="efiber" value="0" min="0" style="width:70px">
      </div>
      <button class="red" onclick="zeroFiber()">Zero fiber</button>
      <button class="red" onclick="pruneFiber()">Prune fiber</button>
      <button class="green" onclick="addFiber()">Add fiber</button>
      <div class="trace" id="editresult"></div>
      <h3>Save / Load (the machine is bytes)</h3>
      <button onclick="saveMachine()">Save to file</button>
      <button onclick="loadMachine()">Load from file</button>
      <input type="file" id="loadfile" accept=".json" style="display:none">
      <div class="trace" id="saveload"></div>
    </div>
  </div>

  <div class="foot">Nautilus · pure JS, no network, no dependencies · a training-method / machine proof of concept, not a product.</div>
</div>

<script>
// ---- the Nautilus machine, inlined. MODEL is the contract: any structure saved by
// engine.StructureEngine.save_structure (with vocab+window) plugs in here. ----
let MODEL = __MODEL__;
let state = MODEL;  // the single source of truth: {window, vocab, bias, dist, rounds}

// dimension accessors read from the LIVE state, so a loaded machine with a different
// vocab/window/fan-out just works — the demo is a model-agnostic environment.
function curV() { return state.vocab.length; }
function curW() { return state.window; }
function curC() { return state.bias.length; }

// ---- forward pass (JS reimplementation) ----
function forwardLogits(x) {
  const C = curC();
  let F = [];
  let logits = new Array(C).fill(0);
  for (let r = 0; r < state.rounds.length; r++) {
    const Wr = state.rounds[r].W, Vr = state.rounds[r].V, br = state.rounds[r].b;
    const nIn = Wr.length, nFib = Wr[0].length;
    const inp = x.concat(F.flat());
    const A = new Array(nFib).fill(0);
    for (let j = 0; j < nFib; j++) {
      let z = br[j];
      for (let i = 0; i < nIn; i++) z += inp[i] * Wr[i][j];
      A[j] = Math.max(0, z);
    }
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let j = 0; j < nFib; j++) s += A[j] * Vr[j][c];
      logits[c] += s;
    }
    F.push(A);
  }
  for (let c = 0; c < C; c++) logits[c] += state.bias[c];
  return logits;
}

function oneHot(prompt) {
  const V = curV(), W = curW();
  const x = new Array(W * V).fill(0);
  const tail = prompt.slice(-W);
  for (let k = 0; k < tail.length; k++) {
    const ci = state.vocab.indexOf(tail[k]);
    if (ci >= 0) x[k * V + ci] = 1;
  }
  return x;
}

function sample(logits, temp) {
  const max = Math.max(...logits);
  const e = logits.map(v => Math.exp((v - max) / Math.max(temp, 1e-6)));
  const sum = e.reduce((a, b) => a + b, 0);
  const p = e.map(v => v / sum);
  let r = Math.random(), acc = 0;
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
  return p.length - 1;
}

function predictChar(x) {
  const logits = forwardLogits(x);
  return { ci: logits.indexOf(Math.max(...logits)), logits };
}

// ---- OBSERVE ----
function renderStats() {
  const V = curV(), W = curW();
  const nFib = state.dist.length;
  const syn = state.rounds.reduce((a, r) => a + r.W.length * r.W[0].length, 0);
  document.getElementById('stats').innerHTML =
    '<span class="stat">rounds: ' + state.rounds.length + '</span>' +
    '<span class="stat">fibers: ' + nFib + '</span>' +
    '<span class="stat">vocab: ' + V + '</span>' +
    '<span class="stat">window: ' + W + '</span>' +
    '<span class="stat">synapses: ' + syn + '</span>';
}
function renderGrowth() {
  let html = '';
  let base = 0;
  for (let r = 0; r < state.rounds.length; r++) {
    const n = state.rounds[r].W[0].length;
    const dists = state.dist.slice(base, base + n);
    const meanD = (dists.reduce((a, b) => a + b, 0) / n).toFixed(2);
    html += '<div class="fiber"><span class="h">round ' + r + '</span> — ' + n + ' fibers, mean distance-from-axiom <span class="d">' + meanD + '</span></div>';
    base += n;
  }
  document.getElementById('growth').innerHTML = html;
}
function showFiber() {
  const r = parseInt(document.getElementById('fround').value);
  const j = parseInt(document.getElementById('ffiber').value);
  const Wr = state.rounds[r], Vr = state.rounds[r], br = state.rounds[r].b;
  if (!Wr || j >= Wr.V.length) { document.getElementById('fiber').innerHTML = '<div class="trace">invalid fiber</div>'; return; }
  let srcs = [];
  for (let i = 0; i < Wr.W.length; i++) {
    if (Wr.W[i][j] !== 0) srcs.push('<span class="src">' + (i < curW()*curV() ? 'input' : 'fiber') + '[' + i + ']</span>=' + Wr.W[i][j].toFixed(2));
  }
  const ro = Vr[j].map((v, c) => v !== 0 ? 'c' + c + ':' + v.toFixed(2) : null).filter(Boolean).join(', ');
  // distance: global index of this fiber
  let base = 0, gidx = -1;
  for (let rr = 0; rr <= r; rr++) { if (rr === r) gidx = base + j; base += state.rounds[rr].W[0].length; }
  const dist = state.dist[gidx] !== undefined ? state.dist[gidx].toFixed(2) : '?';
  document.getElementById('fiber').innerHTML =
    '<div class="fiber"><span class="h">fiber(' + r + ',' + j + ')</span> dist=<span class="d">' + dist + '</span> bias=' + br[j].toFixed(2) + '<br>' +
    '<span class="h">sources:</span> ' + (srcs.join(', ') || 'none') + '<br>' +
    '<span class="h">readout:</span> <span class="ro">' + (ro || 'all zero') + '</span></div>';
}

// ---- TRACE ----
function trace() {
  const prompt = document.getElementById('tprompt').value;
  const x = oneHot(prompt);
  const { ci, logits } = predictChar(x);
  const pred = state.vocab[ci];
  const last = state.rounds.length - 1;
  const Vr = state.rounds[last].V, Wr = state.rounds[last].W;
  const D = curW() * curV();
  let F = [];
  for (let r = 0; r < state.rounds.length; r++) {
    const inp = x.concat(F.flat());
    const A = [];
    for (let j = 0; j < state.rounds[r].W[0].length; j++) {
      let z = state.rounds[r].b[j];
      for (let i = 0; i < state.rounds[r].W.length; i++) z += inp[i] * state.rounds[r].W[i][j];
      A.push(Math.max(0, z));
    }
    F.push(A);
  }
  const Alast = F[last];
  let kf = 0, best = -Infinity;
  for (let j = 0; j < Alast.length; j++) { const s = Alast[j] * Vr[j][ci]; if (s > best) { best = s; kf = j; } }
  const inp = x.concat(F.slice(0, last).flat());
  let s = 0, best2 = -Infinity;
  for (let i = 0; i < Wr.length; i++) { const v = Math.abs(Wr[i][kf]) * Math.abs(inp[i]); if (v > best2) { best2 = v; s = i; } }
  const steps = ['fiber(r' + last + ',#' + kf + ')'];
  for (let hop = 0; hop < 12; hop++) {
    if (s < D) { steps.push('input[' + s + '] (char \'' + state.vocab[s % curV()] + '\')'); break; }
    const j = s - D;
    steps.push('fiber(#' + j + ')');
    s = s % D;
    break;
  }
  document.getElementById('traceout').textContent = "predicted next char: '" + pred + "'";
  document.getElementById('tracesteps').textContent = 'least-path-of-resistance: ' + steps.join(' <- ');
}

// ---- EDIT ----
function result(msg) {
  document.getElementById('editresult').textContent = msg;
  renderStats(); renderGrowth();
}
function zeroFiber() {
  const r = parseInt(document.getElementById('eround').value);
  const j = parseInt(document.getElementById('efiber').value);
  if (!state.rounds[r] || j >= state.rounds[r].V.length) { result('invalid fiber'); return; }
  state.rounds[r].V[j] = state.rounds[r].V[j].map(() => 0);
  result('zeroed fiber(' + r + ',' + j + ')');
}
function pruneFiber() {
  const r = parseInt(document.getElementById('eround').value);
  const j = parseInt(document.getElementById('efiber').value);
  const Wr = state.rounds[r];
  if (!Wr || j >= Wr.V.length) { result('invalid fiber'); return; }
  let base = 0, gidx = -1;
  for (let rr = 0; rr <= r; rr++) { if (rr === r) gidx = base + j; base += state.rounds[rr].W[0].length; }
  for (let rr = r + 1; rr < state.rounds.length; rr++) {
    const pos = curW() * curV() + gidx;
    if (pos < state.rounds[rr].W.length && state.rounds[rr].W[pos].some(v => v !== 0)) {
      result('REFUSED: fiber(' + r + ',' + j + ') is read by a later round (append-only)');
      return;
    }
  }
  Wr.W = Wr.W.map(col => col.filter((_, idx) => idx !== j));
  Wr.V.splice(j, 1); Wr.b.splice(j, 1);
  state.dist.splice(gidx, 1);
  result('pruned fiber(' + r + ',' + j + ')');
}
function addFiber() {
  const r = parseInt(document.getElementById('eround').value);
  const Wr = state.rounds[r];
  if (!Wr) { result('invalid round'); return; }
  const C = curC();
  const col = new Array(Wr.W.length).fill(0);
  col[0] = 1;
  for (let i = 0; i < Wr.W.length; i++) Wr.W[i].push(col[i]);
  Wr.V.push(new Array(C).fill(0));
  Wr.b.push(0);
  state.dist.push(1);
  result('added fiber to round ' + r);
}

// ---- SAVE / LOAD ----
function saveMachine() {
  const blob = new Blob([JSON.stringify({ window: state.window, vocab: state.vocab, rounds: state.rounds, bias: state.bias, dist: state.dist })], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nautilus.json';
  a.click();
  document.getElementById('saveload').textContent = 'saved nautilus.json';
}
function loadMachine() { document.getElementById('loadfile').click(); }
document.getElementById('loadfile').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const d = JSON.parse(reader.result);
      if (!d || !Array.isArray(d.rounds) || !d.vocab || !d.window) {
        document.getElementById('saveload').textContent = 'load failed: not a valid Nautilus machine (needs rounds, vocab, window)';
        throw new Error('invalid');
      }
      state = { rounds: d.rounds, bias: d.bias || [], dist: d.dist || [], vocab: d.vocab, window: d.window };
      document.getElementById('saveload').textContent = 'loaded machine: ' + state.rounds.length + ' rounds, ' + state.dist.length + ' fibers, vocab ' + curV() + ', window ' + curW();
      renderStats(); renderGrowth();
    } catch (err) { if (err.message !== 'invalid') document.getElementById('saveload').textContent = 'load failed: ' + err.message; }
  };
  reader.readAsText(f);
});

// ---- GENERATE ----
function generate() {
  const prompt = document.getElementById('prompt').value;
  const temp = parseFloat(document.getElementById('temp').value);
  let ctx = prompt, out = '', lastChar = '';
  for (let i = 0; i < 40; i++) {
    const x = oneHot(ctx);
    const logits = forwardLogits(x);
    const ci = sample(logits, temp);
    const c = state.vocab[ci];
    out += c; lastChar = c; ctx += c;
  }
  document.getElementById('out').textContent = prompt + out;
  document.getElementById('gentrace').textContent = 'last char: ' + lastChar;
}
function clearOut() { document.getElementById('out').textContent = '(generation appears here)'; document.getElementById('gentrace').textContent = ''; }

// init
renderStats(); renderGrowth();
</script>
</body>
</html>
"""


def build_html(model, baseline_acc, out_path):
    html = (HTML
            .replace("__MODEL__", json.dumps(model))
            .replace("__ACC__", f"{baseline_acc*100:.1f}%")
            .replace("__W__", str(model["window"]))
            .replace("__V__", str(len(model["vocab"]))))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
    return out_path


def main():
    net, vocab, W, Xte, yte = train_compact()
    acc = float((forward_logits(net, Xte).argmax(1) == yte).mean())
    model = export_compact(net, vocab, W)
    out = build_html(model, acc, RESULTS / "nautilus_demo.html")
    print(f"Wrote {out} ({Path(out).stat().st_size/1024:.0f} KB), baseline acc {acc:.3f}")


if __name__ == "__main__":
    main()

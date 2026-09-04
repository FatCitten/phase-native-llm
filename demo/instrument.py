"""demo/instrument.py — the GRAPHICAL Nautilus instrument for human researchers.

A FastAPI server that serves a browser app where a researcher loads any saved
Nautilus structure and SEES it as a graph:
  - nodes = fibers, sized by readout magnitude, colored by distance-from-axiom
  - edges = connections (input->fiber, fiber->fiber)
  - trace = the least-path-of-resistance highlighted
  - metrics = capability-per-synapse + no-forgetting
  - edits = click a fiber -> zero/prune/rewire/add -> graph re-renders live

The graph is drawn with a lightweight JS canvas (no external deps). The server
holds the loaded structure in memory and exposes load/graph/trace/edit/metrics.

Run: python -m demo.instrument  ->  http://localhost:8001
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from demo import metrics, wordlm
from demo.demo import load_sections
from demo.engine import StructureEngine
from experiments.consolidation_rounds import ConsolidatingNet

DEMO = Path(__file__).resolve().parent
app = FastAPI()

# in-memory state: the loaded structure + its data
_state = {"eng": None, "vocab": None, "W": None, "X_old": None, "y_old": None,
          "X_new": None, "y_new": None}


def _default_structure():
    """Train a small word-level Nautilus so the instrument has something to show.
    FAST: small vocab, few epochs — the instrument is for exploring structure, not
    for training a big model (that's the live server's job)."""
    sections = load_sections(DEMO / "corpus.txt")
    vocab = wordlm.build_vocab(list(sections.values()), min_freq=20)  # small vocab
    W = 3
    tokens = wordlm.tokenize(" ".join(sections.values()))
    X, y = wordlm.window_words(tokens, W, vocab)
    # SUBSAMPLE: the instrument is for exploring structure, not training — keep it fast
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), size=min(2000, len(y)), replace=False)
    X, y = X[keep], y[keep]
    Xtr, ytr, Xte, yte = wordlm.split(X, y, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)
    net = ConsolidatingNet(D, C, seed=1)
    for r in range(2):
        net.grow_round(Xtr, ytr, Xte, yte, P=16, epochs=30, tau=0.0)
    _state.update(eng=StructureEngine(net), vocab=vocab, W=W,
                  X_old=Xte, y_old=yte, X_new=Xte, y_new=yte)
    return net


def _graph_data():
    """Build the graph: nodes (fibers) + edges (connections) + trace."""
    eng = _state["eng"]
    net = eng.net
    nodes = []
    edges = []
    base = 0
    for r in range(len(net.frozen_W)):
        W = net.frozen_W[r]
        n = W.shape[1]
        for j in range(n):
            dist = net.dist[base + j]
            ro_mag = float(np.linalg.norm(eng.fiber_readout(r, j)))
            nodes.append({
                "id": f"f{r}_{j}", "round": r, "fiber": j,
                "dist": float(dist), "readout_mag": ro_mag,
            })
            # connections: inputs + earlier fibers
            for kind, src, w in eng.fiber_sources(r, j):
                if kind == "input":
                    edges.append({"from": f"in_{src}", "to": f"f{r}_{j}", "w": float(w)})
                else:
                    edges.append({"from": f"f{src}", "to": f"f{r}_{j}", "w": float(w)})
        base += n
    return {"nodes": nodes, "edges": edges, "D": net.D, "C": net.C,
            "synapses": net.synapses}


class LoadReq(BaseModel):
    path: str | None = None
    data: str | None = None


class TraceReq(BaseModel):
    input: str


class EditReq(BaseModel):
    action: str
    round: int
    fiber: int
    value: float | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (DEMO / "instrument.html").read_text()


@app.post("/load")
def load(req: LoadReq):
    if req.data:
        # load from uploaded JSON
        data = json.loads(req.data)
        D = data.get("D", 0); C = data.get("C", 0)
        net = ConsolidatingNet(D, C, seed=0)
        net.bias = np.array(data["bias"])
        net.dist = list(data["dist"])
        net.frozen_W = [np.array(r["W"]) for r in data["rounds"]]
        net.frozen_V = [np.array(r["V"]) for r in data["rounds"]]
        net.frozen_b = [np.array(r["b"]) for r in data["rounds"]]
        net.synapses = int(data.get("synapses", 0))
        _state["eng"] = StructureEngine(net)
    elif req.path:
        _state["eng"] = StructureEngine.read(req.path)
    else:
        _default_structure()
    return JSONResponse(_graph_data())


@app.post("/graph")
def graph():
    if _state["eng"] is None:
        _default_structure()
    return JSONResponse(_graph_data())


@app.post("/trace")
def trace(req: TraceReq):
    if _state["eng"] is None:
        _default_structure()
    eng = _state["eng"]
    # build one-hot input from the text
    vocab = _state["vocab"] or []
    W = _state["W"] or 4
    V = len(vocab) if vocab else eng.D // W
    x = np.zeros(eng.D)
    tail = req.input[-W:]
    for k, ch in enumerate(tail):
        if ch in vocab:
            x[k * V + vocab.index(ch)] = 1.0
    tr = eng.trace(x)
    return JSONResponse({"pred_class": tr["pred_class"], "steps": tr["steps"]})


@app.post("/edit")
def edit(req: EditReq):
    if _state["eng"] is None:
        _default_structure()
    eng = _state["eng"]
    try:
        if req.action == "zero":
            eng.zero_fiber(req.round, req.fiber)
        elif req.action == "prune":
            eng.prune_fiber(req.round, req.fiber)
        elif req.action == "rewire":
            eng.rewire_fiber(req.round, req.fiber, {0: req.value or 0.0})
        elif req.action == "add":
            eng.add_fiber(req.round, {0: 1.0}, [0.0] * eng.C)
        else:
            return JSONResponse({"error": f"unknown action {req.action}"}, status_code=400)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(_graph_data())


@app.post("/metrics")
def metrics_endpoint():
    if _state["eng"] is None:
        _default_structure()
    eng = _state["eng"]
    cps = metrics.capability_per_synapse(eng.net, _state["X_new"], _state["y_new"])
    old_acc, new_acc = metrics.no_forgetting(eng.net, _state["X_old"], _state["y_old"],
                                             _state["X_new"], _state["y_new"])
    return JSONResponse({"cps": cps, "old_acc": old_acc, "new_acc": new_acc,
                         "synapses": eng.net.synapses})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

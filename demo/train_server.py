"""
demo/train_server.py — the Nautilus live-training website backend.

A FastAPI server that:
  - runs a Nautilus char-LM training loop in a background thread
  - streams live events to the browser over SSE (metrics, structure snapshots,
    agent messages, status)
  - accepts commands from the browser (start/pause/step, chat to the agent,
    agent directives) and from the agent itself (via the same command API)

The "agent" is the LLM (me) driving training. The website lets a human watch
the training live, see the structure being established, and collaborate with
the agent by sending it messages that it can act on.

Run: python -m demo.train_server  ->  serves http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo import charlm
from demo.demo import load_sections
from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits

DEMO = Path(__file__).resolve().parent
RESULTS = Path("results")

app = FastAPI()

# ---------------------------------------------------------------------------
# global training state
# ---------------------------------------------------------------------------
class TrainState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        self.round = 0
        self.epoch = 0
        self.acc = 0.0
        self.loss = 0.0
        self.synapses = 0
        self.fibers = 0
        self.history = []          # list of {round, epoch, acc, loss, synapses}
        self.net = None
        self.vocab = None
        self.W = 8
        self.Xtr = self.ytr = self.Xte = self.yte = None
        self.D = self.C = 0
        self.agent_messages = []  # chat log
        self.agent_directive = None
        self.structure_snapshot = {}  # latest structure view for the browser
        self.subscribers = []     # asyncio queues for SSE

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running, "paused": self.paused,
                "round": self.round, "epoch": self.epoch,
                "acc": round(self.acc, 4), "loss": round(self.loss, 4),
                "synapses": self.synapses, "fibers": self.fibers,
                "history": self.history[-200:],
                "structure": self.structure_snapshot,
                "agent_messages": self.agent_messages[-50:],
                "agent_directive": self.agent_directive,
            }


state = TrainState()


# ---------------------------------------------------------------------------
# the training loop (runs in a background thread)
# ---------------------------------------------------------------------------
def _train_loop():
    """Run consolidation rounds, emitting events as structure is established."""
    while True:
        with state.lock:
            if not state.running or state.paused:
                time.sleep(0.2)
                continue
            net = state.net
            Xtr, ytr, Xte, yte = state.Xtr, state.ytr, state.Xte, state.yte
            r = state.round
        if net is None:
            time.sleep(0.2)
            continue

        # grow one round
        try:
            res = net.grow_round(Xtr, ytr, Xte, yte, P=48, epochs=400, tau=0.0,
                                 floor=0.05, conn_floor=0.1)
        except Exception as e:
            _emit({"type": "error", "message": str(e)})
            with state.lock:
                state.running = False
            time.sleep(0.5)
            continue

        acc = float((forward_logits(net, Xte).argmax(1) == yte).mean())
        with state.lock:
            state.round = r + 1
            state.acc = acc
            state.loss = res.get("test_acc", 0.0)
            state.synapses = net.synapses
            state.fibers = len(net.dist)
            state.history.append({
                "round": r + 1, "epoch": state.epoch, "acc": round(acc, 4),
                "loss": round(res.get("test_acc", 0.0), 4), "synapses": net.synapses,
            })
            state.structure_snapshot = _structure_view(net)
        _emit({"type": "round", "round": r + 1, "acc": round(acc, 4),
               "synapses": net.synapses, "fibers": len(net.dist)})
        _emit({"type": "structure", "data": state.structure_snapshot})
        time.sleep(0.1)


def _structure_view(net):
    """A compact structure snapshot for the browser (rounds, fibers, distances)."""
    rounds = []
    base = 0
    for r in range(len(net.frozen_W)):
        W = net.frozen_W[r]
        n = W.shape[1]
        dists = [float(net.dist[base + j]) for j in range(n)]
        rounds.append({
            "round": r, "n_fibers": n,
            "cross_edges": int((np.abs(W[net.D:]) > 0).sum()) if W.shape[0] > net.D else 0,
            "mean_dist": round(float(np.mean(dists)), 3) if dists else 0,
            "dists": [round(d, 2) for d in dists[:20]],
        })
        base += n
    return {"rounds": rounds, "total_fibers": base, "synapses": net.synapses}


def _emit(event: dict):
    """Push an event to all SSE subscribers."""
    payload = json.dumps(event)
    for q in list(state.subscribers):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------
@app.get("/events")
async def events(request: Request):
    q = asyncio.Queue()
    state.subscribers.append(q)
    try:
        async def gen():
            # send current snapshot first
            yield f"data: {json.dumps({'type': 'snapshot', 'data': state.snapshot()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    continue
        return StreamingResponse(gen(), media_type="text/event-stream")
    finally:
        if q in state.subscribers:
            state.subscribers.remove(q)


# ---------------------------------------------------------------------------
# command API
# ---------------------------------------------------------------------------
class Command(BaseModel):
    action: str
    payload: dict = {}


@app.post("/command")
async def command(cmd: Command):
    a = cmd.action
    p = cmd.payload or {}
    if a == "start":
        with state.lock:
            if state.net is None:
                _init_data()
            state.running = True
            state.paused = False
        _emit({"type": "status", "running": True, "paused": False})
        return {"ok": True}
    if a == "pause":
        with state.lock:
            state.paused = True
        _emit({"type": "status", "running": state.running, "paused": True})
        return {"ok": True}
    if a == "resume":
        with state.lock:
            state.paused = False
        _emit({"type": "status", "running": state.running, "paused": False})
        return {"ok": True}
    if a == "stop":
        with state.lock:
            state.running = False
        _emit({"type": "status", "running": False, "paused": False})
        return {"ok": True}
    if a == "chat":
        # human -> agent message
        msg = p.get("message", "")
        with state.lock:
            state.agent_messages.append({"role": "human", "text": msg})
        _emit({"type": "chat", "role": "human", "text": msg})
        return {"ok": True}
    if a == "agent_reply":
        # agent -> human message
        msg = p.get("message", "")
        with state.lock:
            state.agent_messages.append({"role": "agent", "text": msg})
        _emit({"type": "chat", "role": "agent", "text": msg})
        return {"ok": True}
    if a == "directive":
        # agent sets a directive (what it's doing)
        d = p.get("text", "")
        with state.lock:
            state.agent_directive = d
        _emit({"type": "directive", "text": d})
        return {"ok": True}
    if a == "get_state":
        return {"ok": True, "state": state.snapshot()}
    return {"ok": False, "error": f"unknown action {a}"}


def _init_data():
    """Load corpus + build initial net (called on first start)."""
    sections = load_sections(DEMO / "corpus.txt")
    corpus_a = sections.get("ALICE", "")
    vocab = charlm.build_vocab([corpus_a])
    W = 8
    Xa, ya = charlm.window(corpus_a, W, vocab)
    Xtr, ytr, Xte, yte = charlm.split_windows(Xa, ya, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)
    net = ConsolidatingNet(D, C, seed=1)
    state.net = net
    state.vocab = vocab
    state.W = W
    state.Xtr, state.ytr, state.Xte, state.yte = Xtr, ytr, Xte, yte
    state.D, state.C = D, C
    state.round = 0
    state.history = []


# ---------------------------------------------------------------------------
# static dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (DEMO / "dashboard.html").read_text()


# start the training thread on import
_thread = threading.Thread(target=_train_loop, daemon=True)
_thread.start()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

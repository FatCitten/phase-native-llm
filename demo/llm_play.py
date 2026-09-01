"""
demo/llm_play.py — point an existing LLM at the Nautilus StructureEngine and let it build.

The repo's thesis (MEMORY-AGENT.md): "the LLM is the sighted driver." This gives a real
tool-capable LLM the StructureEngine's tools — observe_growth, fiber_profile, trace,
evaluate, and the edits (zero_fiber, prune_fiber, rewire_fiber, add_fiber, set_readout) —
and lets it explore the structure a char-LM established, then produce its "brain child":
an edited structure plus a report of what it did and why.

The LLM drives the loop via OpenAI function-calling (works with Ollama Cloud or a
local server). It observes, traces, edits, re-evaluates, and finally calls
final_answer with its report. The run is checkpointed after every edit, so a killed
session or reboot loses nothing — the structure (the Nautilus machine) is saved to
disk and can be resumed with --resume.

Pure numpy + stdlib. Run: python -m demo.llm_play [--model glm-5.3-flash]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import charlm, engine, visualizer
from demo.demo import load_sections
from phase_native.ollama_agent import OllamaClient

DEMO = Path(__file__).resolve().parent
LOG_PATH = Path("results") / "llm_play_progress.log"

SYSTEM_PROMPT = """You are a scientist exploring a neural architecture called "consolidation"
(project name: NAUTILUS): a network that grows structure in waves — overproduce candidate
fibers, prune the void, freeze survivors as axioms, then grow further out. Every fiber has
a distance-from-axiom, a set of incoming connections (inputs + earlier fibers), and a
readout to output classes.

You have tools to OBSERVE the structure (growth log, fiber profiles), TRACE a prediction
down to its inputs, EVALUATE accuracy, and EDIT the structure (zero/prune/rewire/add
fibers, set readouts). The model is a tiny char-level next-char predictor on a public-domain
text corpus.

Your task: explore the structure, understand how it works, and create your "brain child" —
a modified structure that you believe is better (higher accuracy, or more interpretable, or
more efficient). Use the tools to:
1. OBSERVE the growth log and fiber profiles to understand the structure.
2. TRACE a few predictions to see how the network reasons.
3. EVALUATE the baseline accuracy.
4. EDIT the structure surgically (zero weak fibers, prune leaves, rewire, add fibers).
5. RE-EVALUATE after each edit to see the effect.
6. When done, call final_answer with a report of what you did, what you found, and what
   your brain child is.

Be honest and scientific. If an edit doesn't help, say so. The structure is append-only:
you cannot prune a fiber that later fibers depend on (the tool will refuse). Prefer
zero_fiber to disable a fiber's output without breaking the graph. Work with the tools —
do not guess. Call final_answer only when you are done exploring."""


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)},
    }}


def build_tools():
    return [
        _tool("observe_growth", "Return the growth log: every round's fibers, their "
              "distance-from-axiom, input/fiber source counts, and cross-edges.", {}),
        _tool("fiber_profile", "Return the full profile of one fiber: sources, readout, "
              "distance, bias.", {"round_idx": {"type": "integer"},
                                  "fiber_idx": {"type": "integer"}},
              ["round_idx", "fiber_idx"]),
        _tool("trace", "Trace the least-path-of-resistance for a test sample index, "
              "showing weights and activations from output to inputs.",
              {"x_index": {"type": "integer"}}, ["x_index"]),
        _tool("evaluate", "Return the current test accuracy of the (possibly edited) net.",
              {}),
        _tool("zero_fiber", "Disable a fiber's output contribution (zero its readout) "
              "without breaking the structure. Safe for any fiber.",
              {"round_idx": {"type": "integer"}, "fiber_idx": {"type": "integer"}},
              ["round_idx", "fiber_idx"]),
        _tool("prune_fiber", "Remove a fiber entirely. Refuses if a later fiber depends "
              "on it (append-only).", {"round_idx": {"type": "integer"},
                                       "fiber_idx": {"type": "integer"}},
              ["round_idx", "fiber_idx"]),
        _tool("rewire_fiber", "Replace a fiber's incoming weights. sources is a dict "
              "{source_idx: weight} where source_idx < D is an input, >= D is a fiber.",
              {"round_idx": {"type": "integer"}, "fiber_idx": {"type": "integer"},
               "sources": {"type": "object"}}, ["round_idx", "fiber_idx", "sources"]),
        _tool("add_fiber", "Add a new fiber to a round. sources is {source_idx: weight}; "
              "readout is a list of C values.", {"round_idx": {"type": "integer"},
              "sources": {"type": "object"}, "readout": {"type": "array",
              "items": {"type": "number"}}}, ["round_idx", "sources", "readout"]),
        _tool("set_readout", "Set one readout weight of a fiber (its contribution to one "
              "output class).", {"round_idx": {"type": "integer"},
              "fiber_idx": {"type": "integer"}, "class_idx": {"type": "integer"},
              "value": {"type": "number"}}, ["round_idx", "fiber_idx", "class_idx", "value"]),
        _tool("final_answer", "Submit your brain child: a report of what you did, what you "
              "found, and the structure you created.", {"report": {"type": "string"}},
              ["report"]),
    ]


class EngineExecutor:
    def __init__(self, eng, Xte, yte, vocab):
        self.eng = eng
        self.Xte = Xte
        self.yte = yte
        self.vocab = vocab
        self.schemas = build_tools()
        self.calls = 0
        self.viz = visualizer.NautilusVisualizer(eng)

    @staticmethod
    def _safe(obj):
        if isinstance(obj, dict):
            return {k: EngineExecutor._safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [EngineExecutor._safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return EngineExecutor._safe(obj.tolist())
        return obj

    @staticmethod
    def _dict_arg(v):
        """Normalize a 'sources' style argument that may arrive as a JSON string."""
        if isinstance(v, str):
            v = json.loads(v) if v.strip() else {}
        return v or {}

    @staticmethod
    def _list_arg(v):
        """Normalize a 'readout' style argument that may arrive as a JSON string."""
        if isinstance(v, str):
            v = json.loads(v) if v.strip() else []
        return list(v or [])

    def _check_fiber(self, round_idx, fiber_idx):
        net = self.eng.net
        if not (0 <= round_idx < len(net.frozen_W)):
            return f"round_idx {round_idx} out of range (0..{len(net.frozen_W)-1})"
        n = net.frozen_W[round_idx].shape[1]
        if not (0 <= fiber_idx < n):
            return f"fiber_idx {fiber_idx} out of range for round {round_idx} (0..{n-1})"
        return None

    def summary(self):
        return EngineExecutor._safe({
            "tool_calls": self.calls,
            "n_rounds": len(self.eng.net.frozen_W),
            "synapses": self.eng.net.synapses,
            "acc": round(self.eng.evaluate(self.Xte, self.yte), 4),
        })

    def dispatch(self, name, args):
        self.calls += 1
        eng = self.eng
        if name == "observe_growth":
            # the LLM sees the SAME canonical view the human visualizer renders
            return {"view": self.viz.view(detail="summary"),
                    "llm_text": self.viz.to_llm(detail="summary")}
        if name == "fiber_profile":
            err = self._check_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            if err:
                return {"ok": False, "error": err}
            return self.eng.fiber_profile(int(args["round_idx"]), int(args["fiber_idx"]))
        if name == "trace":
            i = int(args["x_index"])
            if not (0 <= i < len(self.Xte)):
                return {"ok": False, "error": f"x_index {i} out of range (0..{len(self.Xte)-1})"}
            tr = eng.trace(self.Xte[i])
            return {"pred_class": tr["pred_class"],
                    "true_char": self.vocab[self.yte[i]],
                    "pred_char": self.vocab[tr["pred_class"]],
                    "steps": [{"node": str(s["node"]), "weight": round(s.get("weight", 0), 4),
                               "activation": round(s.get("activation", 0), 4)}
                              for s in tr["steps"]]}
        if name == "evaluate":
            return {"acc": round(eng.evaluate(self.Xte, self.yte), 4),
                    "synapses": eng.net.synapses}
        if name == "zero_fiber":
            err = self._check_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            if err:
                return {"ok": False, "error": err}
            eng.zero_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            return {"ok": True, "acc": round(eng.evaluate(self.Xte, self.yte), 4)}
        if name == "prune_fiber":
            err = self._check_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            if err:
                return {"ok": False, "error": err}
            try:
                eng.prune_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
                return {"ok": True, "acc": round(eng.evaluate(self.Xte, self.yte), 4)}
            except ValueError as e:
                return {"ok": False, "error": str(e)}
        if name == "rewire_fiber":
            err = self._check_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            if err:
                return {"ok": False, "error": err}
            eng.rewire_fiber(int(args["round_idx"]), int(args["fiber_idx"]),
                             {int(k): float(v) for k, v in self._dict_arg(args.get("sources")).items()})
            return {"ok": True, "acc": round(eng.evaluate(self.Xte, self.yte), 4)}
        if name == "add_fiber":
            err = self._check_fiber(int(args["round_idx"]), 0)
            if err:
                return {"ok": False, "error": err}
            eng.add_fiber(int(args["round_idx"]),
                          {int(k): float(v) for k, v in self._dict_arg(args.get("sources")).items()},
                          [float(x) for x in self._list_arg(args.get("readout"))])
            return {"ok": True, "acc": round(eng.evaluate(self.Xte, self.yte), 4)}
        if name == "set_readout":
            err = self._check_fiber(int(args["round_idx"]), int(args["fiber_idx"]))
            if err:
                return {"ok": False, "error": err}
            eng.set_readout(int(args["round_idx"]), int(args["fiber_idx"]),
                            int(args["class_idx"]), float(args["value"]))
            return {"ok": True, "acc": round(eng.evaluate(self.Xte, self.yte), 4)}
        if name == "final_answer":
            return {"ok": True}
        return {"error": f"unknown tool {name}"}


def run(eng, Xte, yte, vocab, model, client, max_iters=60, ex=None):
    if ex is None:
        ex = EngineExecutor(eng, Xte, yte, vocab)
    tools = ex.schemas
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Explore the structure and create your brain "
                 "child. Start by observing the growth log."}]
    report = None
    it = 0
    Path("results").mkdir(exist_ok=True)
    for it in range(1, max_iters + 1):
        resp = client.chat(model, messages, tools)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            report = msg.get("content")
            break
        stop = False
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if name == "final_answer":
                report = args.get("report", "")
                stop = True
                messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                                 "content": json.dumps({"ok": True})})
            else:
                out = json.dumps(EngineExecutor._safe(ex.dispatch(name, args)))
                messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                                 "content": out})
                # incremental progress + checkpoint so nothing is lost on interruption
                line = json.dumps({"it": it, "tool": name, "args": EngineExecutor._safe(args),
                                   "result": json.loads(out)})
                with open(LOG_PATH, "a") as f:
                    f.write(line + "\n")
                if name != "evaluate":
                    _save_checkpoint(eng, Xte, yte, vocab, ex, it)
        if stop:
            break
    return report, ex.calls, it


def _save_checkpoint(eng, Xte, yte, vocab, ex, it):
    """Save the current edited structure so a killed session isn't lost."""
    data = {
        "iteration": it, "tool_calls": ex.calls,
        "acc": round(eng.evaluate(Xte, yte), 4),
        "synapses": eng.net.synapses,
        "rounds": [{"W": w.tolist(), "V": v.tolist(), "b": b.tolist()}
                   for w, v, b in zip(eng.net.frozen_W, eng.net.frozen_V, eng.net.frozen_b)],
        "dist": [float(d) for d in eng.net.dist],
        "bias": eng.net.bias.tolist(),
    }
    Path("results/brain_child_checkpoint.json").write_text(json.dumps(data))


def load_checkpoint(D, C):
    """Rebuild a ConsolidatingNet from results/brain_child_checkpoint.json, if present."""
    ck = Path("results/brain_child_checkpoint.json")
    if not ck.exists():
        return None, 0, 0
    data = json.loads(ck.read_text())
    net = __import__("experiments.consolidation_rounds", fromlist=["ConsolidatingNet"]).ConsolidatingNet(D, C, seed=1)
    net.frozen_W = [np.array(r["W"]) for r in data["rounds"]]
    net.frozen_V = [np.array(r["V"]) for r in data["rounds"]]
    net.frozen_b = [np.array(r["b"]) for r in data["rounds"]]
    net.dist = list(data["dist"])
    net.bias = np.array(data["bias"])
    net.synapses = int(data.get("synapses", 0))
    return net, data["iteration"], data["tool_calls"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--local", action="store_true", help="use local Ollama (no key)")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint")
    args = ap.parse_args()

    print("=" * 72)
    print("LLM PLAYS WITH THE STRUCTURE — the LLM is the sighted driver")
    print("=" * 72)
    print(f"model: {args.model}  local: {args.local}")

    sections = load_sections(DEMO / "corpus.txt")
    corpus_a = sections.get("ALICE", "")
    vocab = charlm.build_vocab([corpus_a])
    W = 16
    Xa, ya = charlm.window(corpus_a, W, vocab)
    Xtr, ytr, Xte, yte = charlm.split_windows(Xa, ya, frac=0.8, seed=1)
    D = W * len(vocab); C = len(vocab)

    resumed = False
    if args.resume:
        net, it0, calls0 = load_checkpoint(D, C)
        if net is not None:
            eng = engine.StructureEngine(net)
            resumed = True
            print(f"\nRESUMED from checkpoint (iteration {it0}, {calls0} tool calls)")
    if not resumed:
        from experiments.consolidation_rounds import ConsolidatingNet
        net = ConsolidatingNet(D, C, seed=1)
        for r in range(4):
            net.grow_round(Xtr, ytr, Xte, yte, P=48, epochs=400, tau=0.0,
                           floor=0.05, conn_floor=0.1)
        eng = engine.StructureEngine(net)

    acc0 = eng.evaluate(Xte, yte)
    print(f"\nbaseline: acc={acc0:.3f}  synapses={eng.net.synapses}  "
          f"rounds={len(eng.net.frozen_W)}")

    if args.local:
        client = OllamaClient(host="http://localhost:11434", timeout=300)
    else:
        client = OllamaClient(timeout=300)
    print(f"\nletting the LLM explore (max {args.max_iters} tool iterations)...\n")
    t0 = time.time()
    report, calls, iters = run(eng, Xte, yte, vocab, args.model, client, args.max_iters)
    dt = time.time() - t0

    acc1 = eng.evaluate(Xte, yte)
    print(f"\n{'='*72}")
    print(f"LLM made {calls} tool calls over {iters} iterations in {dt:.0f}s")
    print(f"accuracy: {acc0:.3f} -> {acc1:.3f}  (Δ {acc1-acc0:+.3f})  "
          f"synapses: {eng.net.synapses}")
    print(f"{'='*72}")
    print("\nTHE LLM'S BRAIN CHILD (its report):\n")
    print(report if report else "(no report — model did not call final_answer)")

    # SAVE the machine: the whole edited structure, loadable by engine.load_structure
    eng.save_structure("results/brain_child.json")
    out = {
        "model": args.model, "baseline_acc": acc0, "final_acc": acc1,
        "tool_calls": calls, "iters": iters, "report": report,
        "structure_file": "results/brain_child.json",
    }
    (Path("results") / "brain_child_meta.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved the Nautilus structure to results/brain_child.json (loadable with "
          f"engine.StructureEngine.read(...)) and results/brain_child_meta.json")


if __name__ == "__main__":
    main()

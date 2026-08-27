"""
Run the phase-memory agent live on an Ollama model (Cloud or local).

The LLM drives the same memory tools as the Claude path: it recalls jumps, computes on miss
(costly `step`s), and solidifies results — so `step`s should fall and memory hits rise as the
memory fills across episodes with recurring start nodes.

Setup (on a machine where Ollama is reachable — this sandbox's proxy blocks ollama.com):
  Cloud:  export OLLAMA_API_KEY=...            # from ollama.com; host defaults to https://ollama.com
  Local:  export OLLAMA_HOST=http://localhost:11434   # no key needed; `ollama pull <model>` first
Pick a tool-capable model (e.g. gpt-oss:120b, gpt-oss:20b, qwen3, llama3.3, qwen2.5).

  python experiments/memory_agent_ollama.py --model gpt-oss:120b --n 6 --max-k 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phase_native.domain import RelationGraph
from phase_native.memory import PhaseNuggetMemory
from phase_native.ollama_agent import OllamaClient, run_agent_ollama
from phase_native.tools import MemoryToolExecutor

RESULTS = Path("results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "https://ollama.com"))
    ap.add_argument("--n", type=int, default=6, help="episodes")
    ap.add_argument("--max-k", type=int, default=15, help="max hops per episode")
    ap.add_argument("--nodes", type=int, default=64)
    ap.add_argument("--anchors", type=int, default=3, help="recurring start nodes")
    ap.add_argument("--max-iters", type=int, default=30)
    args = ap.parse_args()

    key = os.environ.get("OLLAMA_API_KEY")
    if "localhost" not in args.host and "127.0.0.1" not in args.host and not key:
        print("[ollama] No OLLAMA_API_KEY set for a cloud host. Export it (from ollama.com) "
              "or point --host at a local server.")
        return

    client = OllamaClient(host=args.host, api_key=key)
    g = RelationGraph(n_nodes=args.nodes, seed=7)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=6144)
    ex = MemoryToolExecutor(graph=g, memory=mem)

    rng = np.random.default_rng(3)
    anchors = rng.integers(0, args.nodes, size=args.anchors)  # recur -> memory should help
    print(f"[ollama] host={args.host} model={args.model} n={args.n} max_k={args.max_k} "
          f"anchors={sorted(set(int(a) for a in anchors))}")

    rows = []
    for i in range(args.n):
        s = int(rng.choice(anchors))
        k = int(rng.integers(1, args.max_k + 1))
        t0 = time.time()
        try:
            res = run_agent_ollama(s, k, ex, model=args.model, client=client, max_iters=args.max_iters)
        except urllib.error.HTTPError as e:
            print(f"  ep{i}: HTTP {e.code} from Ollama — check the model name is pulled/available "
                  f"and the key/host are correct. ({e.reason})")
            return
        except urllib.error.URLError as e:
            print(f"  ep{i}: cannot reach {args.host} ({e.reason}). On this sandbox ollama.com is "
                  f"proxy-blocked; run on your machine.")
            return
        ok = res.answer == g.truth_pow(s, k)
        rows.append({"i": i, "s": s, "k": k, "ok": bool(ok), "steps": res.steps,
                     "hits": res.hits, "recalls": res.recalls, "writes": res.writes,
                     "out_tokens": res.output_tokens, "iters": res.iters})
        print(f"  ep{i:2d} reach({s:3d},{k:4d}) -> {res.answer} truth={g.truth_pow(s,k)} ok={ok} | "
              f"steps={res.steps:3d} hits={res.hits:2d}/{res.recalls:2d} writes={res.writes:2d} "
              f"iters={res.iters:2d} out_tok={res.output_tokens} {time.time()-t0:.0f}s")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "memory_agent_ollama.json").write_text(json.dumps(
        {"model": args.model, "host": args.host, "episodes": rows,
         "memory": mem.stats()}, indent=2))
    if rows:
        acc = sum(r["ok"] for r in rows) / len(rows)
        print(f"[ollama] accuracy={acc:.2f}; step-calls and output tokens should trend DOWN as "
              f"the memory fills. Saved results/memory_agent_ollama.json")


if __name__ == "__main__":
    main()

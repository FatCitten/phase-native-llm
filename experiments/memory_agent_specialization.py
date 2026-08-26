"""
Specialization: compute-to-answer falls as the LLM solidifies established results.

OFFLINE (always runs, reproducible, no API): the ScriptedDriver answers a stream of
reach(s,k) problems through the phase-memory tools, with memory ON vs OFF. Memory-OFF
re-derives every path (~constant, high compute). Memory-ON solidifies jump nuggets and
increasingly answers by O(1) recall -> compute per query collapses while accuracy holds.

LIVE (--live, needs Anthropic credentials): the real LLM drives the same tools
(phase_native.agent.run_agent). Reports output tokens + step calls per episode, falling as
the memory fills. Where credentials are absent it prints the exact command and skips.

Run:
  python experiments/memory_agent_specialization.py                 # offline proof
  python experiments/memory_agent_specialization.py --live --n 8    # + live loop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_native.domain import QueryStream, RelationGraph
from phase_native.driver import ScriptedDriver
from phase_native.memory import PhaseNuggetMemory
from phase_native.tools import MemoryToolExecutor

RESULTS = Path("results")


def run_offline(n_queries=200, reps=6144, seed=0):
    def run(with_mem):
        g = RelationGraph(n_nodes=256, seed=7)
        mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=reps) if with_mem else None
        ex = MemoryToolExecutor(graph=g, memory=mem)
        drv = ScriptedDriver(ex)
        qs = QueryStream(g, max_k=1023, n_queries=n_queries, seed=seed)
        steps, correct = [], 0
        for s, k in qs:
            g.reset_counter()
            ans = drv.solve(s, k)
            steps.append(g.steps_taken)
            correct += ans == qs.truth(s, k)
        return np.array(steps), correct / n_queries, ex, (mem.stats() if mem else None)

    off_steps, off_acc, _, _ = run(False)
    on_steps, on_acc, ex, st = run(True)
    return {
        "off_steps": off_steps,
        "off_acc": off_acc,
        "on_steps": on_steps,
        "on_acc": on_acc,
        "hits": ex.hits,
        "recalls": ex.recalls,
        "stats": st,
    }


def plot_offline(r, path):
    off, on = r["off_steps"], r["on_steps"]
    q = np.arange(1, len(off) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(q, np.cumsum(off) / q, label=f"memory OFF (acc {r['off_acc']:.2f})")
    ax1.plot(q, np.cumsum(on) / q, label=f"memory ON  (acc {r['on_acc']:.2f})")
    ax1.set_xlabel("queries seen")
    ax1.set_ylabel("cumulative avg compute (steps/query)")
    ax1.set_title("Compute-to-answer collapses as structure is solidified")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    tot_off, tot_on = int(off.sum()), int(on.sum())
    ax2.bar(["OFF", "ON"], [tot_off, max(tot_on, 1)], color=["#b0653c", "#3c7fb0"])
    ax2.set_yscale("log")
    ax2.set_ylabel("total compute (steps, log)")
    ax2.set_title(f"Total compute: {tot_off} -> {tot_on}  ({tot_off/max(tot_on,1):.0f}x less)")
    for i, v in enumerate([tot_off, tot_on]):
        ax2.text(i, max(v, 1), str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def run_live(n=8, model="claude-opus-5", max_k=1023):
    from phase_native.agent import credentials_available, run_agent

    cmd = f"python experiments/memory_agent_specialization.py --live --n {n} --model {model}"
    if not credentials_available():
        print("\n[LIVE] No Anthropic credentials resolvable in this environment.")
        print("       The LLM-driven demonstrator needs an API key / `ant auth login`.")
        print(f"       Where creds exist, run:  {cmd}")
        return None

    import anthropic

    from phase_native.agent import SYSTEM_PROMPT

    client = anthropic.Anthropic()
    g = RelationGraph(n_nodes=256, seed=7)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=6144)
    ex = MemoryToolExecutor(graph=g, memory=mem)

    # cost estimate before spending (also confirms the credential actually works)
    try:
        est = client.messages.count_tokens(
            model=model, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Compute reach(s=0, k=1023)."}],
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, TypeError) as e:
        print(f"\n[LIVE] Credentials present but the API call failed: {type(e).__name__}.")
        print(f"       Where valid creds exist, run:  {cmd}")
        return None
    print(f"\n[LIVE] model={model}  ~{est.input_tokens} input tokens/turn before tools; "
          f"running {n} episodes (each is several turns). Ctrl-C to abort.")

    rng = np.random.default_rng(3)
    anchors = rng.integers(0, 256, size=4)  # few anchors so jumps recur -> memory helps
    rows = []
    for i in range(n):
        s = int(rng.choice(anchors))
        k = int(rng.integers(1, max_k + 1))
        res = run_agent(s, k, ex, model=model, client=client)
        ok = res.answer == g.truth_pow(s, k)
        rows.append({"i": i, "s": s, "k": k, "ok": bool(ok), "steps": res.steps,
                     "hits": res.hits, "out_tokens": res.output_tokens})
        print(f"  ep{i:2d} reach({s:3d},{k:4d}) ok={ok} steps={res.steps:3d} "
              f"hits={res.hits:2d} out_tok={res.output_tokens}")
    print("[LIVE] step calls and output tokens should trend DOWN as the memory fills.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--reps", type=int, default=6144)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--n", type=int, default=8, help="live episodes")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-k", type=int, default=1023, help="max hops per live episode")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    print("=== OFFLINE (scripted driver, reproducible) ===")
    r = run_offline(args.queries, args.reps)
    print(f"memory OFF: total steps={int(r['off_steps'].sum()):7d}  acc={r['off_acc']:.3f}")
    print(f"memory ON : total steps={int(r['on_steps'].sum()):7d}  acc={r['on_acc']:.3f}  "
          f"({int(r['off_steps'].sum())/max(int(r['on_steps'].sum()),1):.0f}x less compute)")
    print(f"recall hits: {r['hits']}/{r['recalls']}   memory: {r['stats']}")
    plot_offline(r, RESULTS / "memory_agent_specialization.png")
    (RESULTS / "memory_agent_specialization.json").write_text(json.dumps({
        "off_total_steps": int(r["off_steps"].sum()), "off_acc": r["off_acc"],
        "on_total_steps": int(r["on_steps"].sum()), "on_acc": r["on_acc"],
        "on_steps_per_query": r["on_steps"].tolist(), "hits": r["hits"],
        "recalls": r["recalls"], "stats": r["stats"],
    }, indent=2))
    print("Saved results/memory_agent_specialization.{png,json}")

    live_rows = run_live(args.n, args.model, args.max_k) if args.live else None
    if live_rows is not None:
        (RESULTS / "memory_agent_live.json").write_text(json.dumps(live_rows, indent=2))
        print("Saved results/memory_agent_live.json")


if __name__ == "__main__":
    main()

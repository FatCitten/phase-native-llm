"""
Self-checking tests (assert/print style). No API, no torch — numpy only.

Run: python tests/test_phase_native.py   ->   exits nonzero if any test fails.
Covers: CRT round-trip, bind/unbind invertibility, associative recall + confidence gate,
exact forget, low-byte serialize, the scripted offline loop (memory cuts compute, accuracy
holds), and the live agent's tool-loop plumbing via a mock Claude client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phase_native import PhaseNuggetMemory, bind, unbind
from phase_native.codebook import CRTValueCodebook, crt_combine, key_vector
from phase_native.compose import compose_reach, edge_cue, recall_chain
from phase_native.domain import QueryStream, RelationGraph
from phase_native.driver import ScriptedDriver
from phase_native.tools import MemoryToolExecutor

_failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _failures.append(name)


def test_crt():
    print("CRT value codebook")
    vb = CRTValueCodebook((8, 9, 5, 7, 11, 13), reps=8)
    check("crt_combine solves x=1(mod3), x=2(mod5) -> 7", crt_combine([1, 2], [3, 5]) == 7)
    check("clean encode/decode round-trips over the range",
          all(vb.decode(vb.encode(v)) == v for v in range(0, vb.capacity, 991)))


def test_ops():
    print("bind / unbind invertibility")
    a = key_vector("A", 64)
    b = key_vector("B", 64)
    recovered = unbind(bind(a, b), a)
    check("unbind(bind(a,b), a) == b", np.allclose(recovered, b, atol=1e-9))


def test_memory():
    print("PhaseNuggetMemory recall / gate / forget / serialize")
    mem = PhaseNuggetMemory()  # dim 2048
    facts = {f"cue_{i}": f"val_{i}" for i in range(40)}
    for c, p in facts.items():
        mem.write(c, p)
    check("all 40 recalled correctly", all(mem.recall(c).payload == p for c, p in facts.items()))
    check("confident hit (>0.45) for a written cue", mem.recall("cue_3").confidence > 0.45)
    check("unwritten cue is a miss (low confidence)", not mem.recall("nope").hit)
    mem.forget("cue_5")
    check("forget removes exactly its cue", not mem.recall("cue_5").hit)
    check("forget leaves others intact",
          sum(mem.recall(c).payload == p for c, p in facts.items() if c != "cue_5") == 39)
    mem2 = PhaseNuggetMemory.from_dict(mem.to_dict())
    check("serialize round-trip preserves recalls",
          all(mem2.recall(c).payload == mem.recall(c).payload for c in facts))


def test_composition():
    print("Compositional multi-hop recall (chain atomic nuggets)")
    g = RelationGraph(n_nodes=64, seed=7, bijective=True)  # light load, ample dim -> ~100%/hop
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=2048)
    for n in range(64):
        mem.write(edge_cue(n), g.step(n))
    deep_ok = all(
        recall_chain(mem, s, d).node == g.truth_pow(s, d)
        for s in range(0, 64, 7) for d in (1, 5, 20, 80)
    )
    check("composes atomic facts into deep answers (depth up to 80)", deep_ok)

    r = recall_chain(mem, 0, 40)
    check("a fully-confident chain reports ok + high min-confidence", r.ok and r.min_confidence > 0.45)

    # compose_reach learns each atomic edge at most once, then answers for free
    g2 = RelationGraph(n_nodes=64, seed=7, bijective=True)
    mem2 = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=2048)
    first = compose_reach(mem2, g2, 3, 200)
    steps_after_warm = g2.steps_taken
    g2.reset_counter()
    second = compose_reach(mem2, g2, 3, 200)  # same query, now fully cached
    check("compose_reach is correct", first.node == g2.truth_pow(3, 200))
    check("re-answering a learned query costs 0 steps", g2.steps_taken == 0)
    check("atomic edges learned at most once (<= n_nodes steps)", steps_after_warm <= 64)


def test_scripted_loop():
    print("Scripted offline loop: memory cuts compute, accuracy holds")

    def run(with_mem):
        g = RelationGraph(n_nodes=256, seed=7)
        mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=6144) if with_mem else None
        ex = MemoryToolExecutor(graph=g, memory=mem)
        drv = ScriptedDriver(ex)
        qs = QueryStream(g, max_k=1023, n_queries=120, seed=0)
        steps, correct = 0, 0
        for s, k in qs:
            g.reset_counter()
            ans = drv.solve(s, k)
            steps += g.steps_taken
            correct += ans == qs.truth(s, k)
        return steps, correct / 120

    off_steps, off_acc = run(False)
    on_steps, on_acc = run(True)
    check("memory-off is correct (brute force)", off_acc > 0.99)
    check("memory-on stays accurate", on_acc > 0.99)
    check(f"memory-on uses >>10x less compute ({off_steps}->{on_steps})", on_steps * 10 < off_steps)


# ---- mock Claude client to exercise the real run_agent tool loop (no API) -------------
class _MockClient:
    """Emits a fixed script of tool calls so agent.run_agent's plumbing is tested offline."""

    def __init__(self):
        self.turn = 0
        self.messages = SimpleNamespace(create=self._create)
        self._script = [
            ("memory_write", {"cue": "jump(node=5,level=0)", "conclusion": "42"}),
            ("memory_recall", {"cue": "jump(node=5,level=0)"}),
            ("step", {"node": 5}),
            ("final_answer", {"node": 99}),
        ]

    def _create(self, **_):
        name, inp = self._script[self.turn]
        self.turn += 1
        block = SimpleNamespace(type="tool_use", id=f"t{self.turn}", name=name, input=inp)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage)


def test_agent_plumbing():
    print("Live-agent tool loop (mock client)")
    from phase_native.agent import run_agent

    g = RelationGraph(n_nodes=256, seed=7)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=512)
    ex = MemoryToolExecutor(graph=g, memory=mem)
    res = run_agent(0, 1, ex, client=_MockClient())
    check("loop terminates on final_answer with the submitted node", res.answer == 99)
    check("write + recall + step were dispatched", res.writes == 1 and res.recalls == 1 and res.steps == 1)
    check("the recall hit the just-written nugget", res.hits == 1)
    check("usage accumulated across turns", res.output_tokens == 20 and res.input_tokens == 40)


class _MockOllama:
    """Mock OpenAI-compatible chat client: scripts tool_calls to exercise run_agent_ollama."""

    def __init__(self):
        self.turn = 0
        self._script = [
            ("memory_write", '{"cue": "jump(node=5,level=0)", "conclusion": "42"}'),
            ("memory_recall", '{"cue": "jump(node=5,level=0)"}'),
            ("step", '{"node": 5}'),
            ("final_answer", '{"node": 99}'),
        ]

    def chat(self, model, messages, tools):
        name, args = self._script[self.turn]
        self.turn += 1
        return {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{self.turn}", "type": "function",
                 "function": {"name": name, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def test_ollama_agent_plumbing():
    print("Ollama-agent tool loop (mock OpenAI-compatible client)")
    from phase_native.ollama_agent import run_agent_ollama, to_openai_tools
    from phase_native.tools import MEMORY_TOOL_SCHEMAS

    conv = to_openai_tools(MEMORY_TOOL_SCHEMAS)
    check("tool schemas convert to OpenAI function shape",
          conv[0]["type"] == "function" and "parameters" in conv[0]["function"])

    g = RelationGraph(n_nodes=256, seed=7)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=512)
    ex = MemoryToolExecutor(graph=g, memory=mem)
    res = run_agent_ollama(0, 1, ex, client=_MockOllama())
    check("loop terminates on final_answer", res.answer == 99)
    check("write + recall + step dispatched via OpenAI tool_calls",
          res.writes == 1 and res.recalls == 1 and res.steps == 1 and res.hits == 1)
    check("usage accumulated (prompt/completion tokens)",
          res.output_tokens == 20 and res.input_tokens == 40)


def main():
    for t in (test_crt, test_ops, test_memory, test_composition, test_scripted_loop,
              test_agent_plumbing, test_ollama_agent_plumbing):
        t()
    print()
    if _failures:
        print(f"*** {len(_failures)} FAILED: {_failures} ***")
        sys.exit(1)
    print("*** ALL TESTS PASSED ***")


if __name__ == "__main__":
    main()

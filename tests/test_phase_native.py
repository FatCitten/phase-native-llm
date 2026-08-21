"""
Self-checking tests (assert/print style, matching analysis/test_crt.py). No API, no torch.

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
    check("crt_combine matches analysis/test_crt.py case", crt_combine([1, 2], [3, 5]) == 7)
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


def main():
    for t in (test_crt, test_ops, test_memory, test_scripted_loop, test_agent_plumbing):
        t()
    print()
    if _failures:
        print(f"*** {len(_failures)} FAILED: {_failures} ***")
        sys.exit(1)
    print("*** ALL TESTS PASSED ***")


if __name__ == "__main__":
    main()

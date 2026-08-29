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


def test_lucid_fuzzy():
    print("Fuzzy verify-before-assert (LucidMemory)")
    from phase_native.lucid_guard import LucidMemory

    lm = LucidMemory(reps=1024)
    lm.commit("the migration must run before deploy because the new column is non-nullable")
    lm.commit("rate limiting is enforced per api key not per ip address")
    v = lm.verify("run the migration prior to deploying, the added column is non nullable")
    check("recalls a paraphrase exact grep would miss",
          v.status == "recalled" and "migration" in str(v.statement))
    check("a recall carries a re-derivable receipt", v.receipt is not None and "residues" in v.receipt)
    u = lm.verify("the frontend uses tailwind for styling")
    check("abstains on a never-stored claim", u.status == "unknown")


def test_consolidation():
    print("Iterative consolidation: inviolate axioms, void-prune, tightening cross-paths")
    from experiments.consolidation_rounds import (
        ConsolidatingNet, fiber_distance, hier_teacher_data)

    # (a) the 'further from axiom' formula, scripted and deterministic
    check("a fiber reading only inputs sits at distance 1",
          abs(fiber_distance(np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])) - 1.0) < 1e-6)
    check("a fiber bundling a distance-2 primitive lands at distance 3",
          abs(fiber_distance(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0])) - 3.0) < 1e-6)
    d_near = fiber_distance(np.array([3.0, 1.0]), np.array([0.0, 2.0]))  # weight mostly on the input
    d_far = fiber_distance(np.array([1.0, 3.0]), np.array([0.0, 2.0]))   # weight mostly on the far src
    check("leaning on further sources monotonically increases distance", d_far > d_near)

    Xtr, ytr, Xte, yte, D, C = hier_teacher_data(N=1500, ntr=1100, seed=0)

    # (b) a short LOOSE loop (tau=0): void pruned, axioms inviolate, structure grows outward
    net = ConsolidatingNet(D, C, seed=1)
    r1 = net.grow_round(Xtr, ytr, Xte, yte, P=16, epochs=300, floor=0.5, conn_floor=0.35, refit=150)
    axiom = net.frozen_W[0].copy()  # the round-1 axioms
    stats = [r1]
    for _ in range(3):
        stats.append(net.grow_round(Xtr, ytr, Xte, yte, P=16, epochs=300, conn_floor=0.35, refit=150))
    check("overproduced candidates are pruned as void (kept < P)", 1 <= r1["kept"] < 16)
    check("round-1 axioms are byte-identical after all later rounds",
          np.array_equal(axiom, net.frozen_W[0]))
    check("the frozen base only ever grows (append-only)",
          len(net.frozen_W) == 4 and len(net.dist) > r1["kept"])
    check("structure grows outward (mean distance-from-axiom rises past round 1)",
          stats[-1]["mean_dist"] > stats[0]["mean_dist"] + 0.05)

    # (c) the TIGHTENING RATIO pulls concept-lines into a sparse cross-path mesh, cutting wiring
    def mini(sched, seed=2):
        nt = ConsolidatingNet(D, C, seed=seed)
        hs = [nt.grow_round(Xtr, ytr, Xte, yte, P=16, epochs=300, tau=t) for t in sched]
        return nt, hs
    tt, th = mini([0.0, 0.3, 0.5, 0.7])
    uu, _ = mini([0.0, 0.0, 0.0, 0.0])
    check("tightening forms cross-paths once a base exists (fiber->fiber edges appear)",
          th[1]["cross_edges"] > 0 and tt.cross_edges > 0)
    check("each new fiber stays sparsely bundled (<= k_par=6 parents)",
          all(s["cross_edges"] <= 6 * s["kept"] for s in th))
    check("survivors meet the tightening ratio (base-mass share tracks the ramp)",
          th[-1]["base_share"] >= 0.4)
    check("the tightened mesh is sparser than the loose one (fewer total edges)",
          tt.cross_edges < uu.cross_edges)
    check("tightening cuts total wiring vs the untightened loop (same seed)",
          tt.synapses < uu.synapses)


def test_multi_teacher():
    print("Multiple teachers on one spine: prune-as-feature, reuse, no-forgetting, LPR trace")
    import experiments.multi_teacher as mt
    from experiments.consolidation_rounds import ConsolidatingNet

    Xtr, Ytr, Xte, Yte, D, C, T = mt.shared_primitive_teachers(T=3, N=900, ntr=680, seed=0)

    # (a) integral magnitude prune (snip loose threads) cuts wiring
    a = ConsolidatingNet(D, C, seed=1); ra = a.grow_round(Xtr, Ytr[:, 0], Xte, Yte[:, 0], P=16, epochs=200, tau=0.5)
    b = ConsolidatingNet(D, C, seed=1); rb = b.grow_round(Xtr, Ytr[:, 0], Xte, Yte[:, 0], P=16, epochs=200, tau=0.5, prune_density=0.5)
    check("snipping loose threads (prune_density) cuts wiring", rb["synapses"] < ra["synapses"])

    # (b) shared spine + isolated branches -> exact no-forgetting + reuse
    spine = mt.build_spine(Xtr, Ytr[:, 0], Xte, Yte[:, 0], D, C, P=16, EP=200)
    sw = len(spine.dist)
    br0 = mt.grow_branch(spine, Xtr, Ytr[:, 0], Xte, Yte[:, 0], D, C, P=16, EP=200, seed=10)
    W0, acc0 = br0.frozen_W[-1].copy(), br0.acc(Yte[:, 0])
    _ = mt.grow_branch(spine, Xtr, Ytr[:, 1], Xte, Yte[:, 1], D, C, P=16, EP=200, seed=11)
    check("adding a new teacher leaves the prior branch byte-identical (no forgetting)",
          np.array_equal(W0, br0.frozen_W[-1]) and br0.acc(Yte[:, 0]) == acc0)
    check("a branch reuses the shared spine (bundles spine fibers via cross-paths)",
          int((np.abs(br0.frozen_W[-1][D:]) > 0).sum()) > 0 and br0.spine_width == sw)

    # (c) least-path-of-resistance trace is a valid output -> ... -> input walk
    path, pred = mt.trace_lpr(br0, spine, Xte[0], 0, D)
    check("LPR trace starts at a branch fiber and ends at an input",
          path[0][0] == "branch" and path[-1][0] == "input" and len(path) >= 2)


def test_spine_growth():
    print("Spine growth: balance gates, append-only growth, shortcut node, grafting two brains")
    import experiments.spine_growth as sg

    # (a) the admission gates
    check("gate_balanced rejects a single dominating parent, accepts a spread block",
          (not sg.gate_balanced(np.array([5.0, 0.02, 0.0]))) and sg.gate_balanced(np.array([1.0, 1.0, 1.0])))
    rng = np.random.default_rng(0); base = rng.normal(0, 1, (60, 3))
    check("subspace-novelty admits a new direction, rejects a near-duplicate",
          sg.is_novel(rng.normal(0, 1, 60), base) and not sg.is_novel(base[:, 0] * 1.001 + 1e-6, base))

    # (b) sequential growth is append-only (a promoted block is never disturbed)
    Xtr, Ytr, Xte, Yte, D, C, T, _ = sg.growing_teachers(T=4, prim=6, k=4, N=800, ntr=600, seed=0)
    g = sg.grow_sequentially(Xtr, Ytr, Xte, Yte, D, C, T, promote=True, P=12, EP=150)
    check("the spine only ever grows (widths non-decreasing)",
          all(g["widths"][i] <= g["widths"][i + 1] for i in range(T - 1)))
    check("later teachers reuse the spine (reuse fraction rises above zero)", max(g["reuses"]) > 0.1)

    # (c) a self-specializing node distills a deep fiber and shortens the path
    sc = sg.distill_shortcut(g["spine"], Xtr, Xte)
    check("shortcut node reproduces the deep fiber (r2 > 0.3) and shortens the path",
          sc["r2"] > 0.3 and sc["dist_saved"] > 0)

    # (d) grafting two brains: sources preserved, cross-brain edges, beats either brain alone
    XAtr, XAte, yAtr, yAte, yBtr, yBte, ycr, ycrte, D3, C3, half = sg.two_brain_teachers(N=900, ntr=680, seed=1)
    def mask(X, s):
        Z = X.copy(); Z[:, half:] = 0.0 if s == "L" else Z[:, half:]; Z[:, :half] = 0.0 if s == "R" else Z[:, :half]; return Z
    A = sg.build_bank(mask(XAtr, "L"), yAtr, mask(XAte, "L"), yAte, D3, C3, seed=2, P=16, EP=200)
    B = sg.build_bank(mask(XAtr, "R"), yBtr, mask(XAte, "R"), yBte, D3, C3, seed=3, P=16, EP=200)
    Asnap = A.Ftr.copy()
    graft, eA, eB = sg.graft_brains(A, B, XAtr, ycr, XAte, ycrte, D3, C3, P=16, EP=200)
    check("grafting preserves both source brains byte-for-byte", np.array_equal(Asnap, A.Ftr))
    check("the graft bundles BOTH brains (cross-brain edges on each side)", eA > 0 and eB > 0)
    accA = sg.solo_acc(A, XAtr, ycr, XAte, ycrte, D3, C3, P=16, EP=200)
    accB = sg.solo_acc(B, XAtr, ycr, XAte, ycrte, D3, C3, P=16, EP=200)
    check("the graft beats what either brain alone can do on the cross task",
          graft.acc(ycrte) > max(accA, accB))


def main():
    for t in (test_crt, test_ops, test_memory, test_composition, test_scripted_loop,
              test_agent_plumbing, test_ollama_agent_plumbing, test_lucid_fuzzy, test_consolidation,
              test_multi_teacher, test_spine_growth):
        t()
    print()
    if _failures:
        print(f"*** {len(_failures)} FAILED: {_failures} ***")
        sys.exit(1)
    print("*** ALL TESTS PASSED ***")


if __name__ == "__main__":
    main()

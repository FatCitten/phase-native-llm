"""
demo/agent_driver.py — the agent (an LLM) that watches and drives Nautilus training.

This is the "me" side of the live-training website. It polls the train_server for
new human chat messages and new training state, and:
  - posts a directive (what the agent is doing) to the dashboard
  - replies to the human's messages via the command API
  - can issue training commands (start/pause/stop) based on what it sees

The agent is driven by an LLM (via Ollama Cloud) that reads the current training
state + the human's latest message and decides what to say/do. This makes the
website a genuine human-LLM collaboration surface.

Run: python -m demo.agent_driver --model glm-5.3-flash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_native.ollama_agent import OllamaClient

BASE = "http://localhost:8000"


def api(action, payload=None):
    body = json.dumps({"action": action, "payload": payload or {}}).encode()
    req = urllib.request.Request(f"{BASE}/command", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def get_state():
    return api("get_state")["state"]


def agent_say(text):
    api("agent_reply", {"message": text})


def agent_directive(text):
    api("directive", {"text": text})


def agent_act(action):
    api(action)


SYSTEM = """You are the Nautilus training agent. You are training a consolidation
neural network (a char-level next-char model) live, and a human is watching the
dashboard and can message you. Your job:
- Read the current training state (round, accuracy, synapses, fibers, structure).
- If the human sent a message, respond helpfully and concisely (1-3 sentences).
- Set a directive describing what you're doing right now.
- Optionally issue a training command (start/pause/resume/stop) if it makes sense.

Be honest and concise. The model is tiny and imperfect; don't oversell it."""


def build_prompt(state, human_msgs):
    lines = ["Current training state:"]
    lines.append(f"  running={state['running']} paused={state['paused']} "
                 f"round={state['round']} acc={state['acc']} "
                 f"synapses={state['synapses']} fibers={state['fibers']}")
    if state["history"]:
        last = state["history"][-1]
        lines.append(f"  last round: acc={last['acc']} synapses={last['synapses']}")
    if state["structure"] and state["structure"].get("rounds"):
        r = state["structure"]["rounds"][-1]
        lines.append(f"  newest round: {r['n_fibers']} fibers, mean_dist={r['mean_dist']}")
    if human_msgs:
        lines.append("\nLatest human message(s):")
        for m in human_msgs[-3:]:
            lines.append(f"  [{m['role']}] {m['text']}")
    lines.append("\nRespond with a JSON object: {\"reply\": \"...\", \"directive\": \"...\", \"action\": \"none|start|pause|resume|stop\"}")
    return "\n".join(lines)


def parse_response(text):
    """Extract the JSON from the model's reply (tolerate prose around it)."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"reply": text.strip()[:200], "directive": "thinking", "action": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    client = OllamaClient(timeout=120)
    print(f"agent_driver: model={args.model} watching {BASE} every {args.interval}s")
    seen = set()

    while True:
        try:
            state = get_state()
            # find new human messages
            new_msgs = [m for m in state["agent_messages"]
                        if m["role"] == "human" and id(m) not in seen]
            # track by (role,text) since ids change across snapshots
            new_msgs = [m for m in state["agent_messages"]
                        if m["role"] == "human" and (m["role"], m["text"]) not in seen]
            for m in new_msgs:
                seen.add((m["role"], m["text"]))

            if new_msgs or state["round"] % 3 == 0:  # respond to messages, or check in periodically
                prompt = build_prompt(state, new_msgs)
                resp = client.chat(args.model, [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ], [])
                text = resp["choices"][0]["message"].get("content", "")
                parsed = parse_response(text)
                if parsed.get("reply"):
                    agent_say(parsed["reply"])
                if parsed.get("directive"):
                    agent_directive(parsed["directive"])
                act = parsed.get("action", "none")
                if act in ("start", "pause", "resume", "stop"):
                    agent_act(act)
                    print(f"  agent action: {act}")
                print(f"  agent: {parsed.get('reply','')[:80]}")
        except Exception as e:
            print(f"  (agent loop: {e})")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

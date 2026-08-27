"""Phase-native memory: an O(1) associative store an LLM builds and calls.

See README.md and FOR_AN_LLM.md for the full story. Core pieces:
  codebook  - CRT value encoding (O(1) decode) + deterministic random keys
  ops       - bind / unbind / superpose / cleanup (elementwise, no gradients)
  memory    - PhaseNuggetMemory: write / recall(O(1)) / forget / serialize
  tools     - Claude tool schemas + executor for the memory
  driver    - Driver protocol + ScriptedDriver (offline test scaffolding)
  agent     - ClaudeDriver + run_agent (the real LLM-driven solution)
  domain    - synthetic recurring-subproblem reasoning task
"""

from .codebook import CRTValueCodebook, crt_combine, key_vector
from .compose import ChainResult, compose_reach, lifted_reach, recall_chain
from .memory import PhaseNuggetMemory, RecallResult
from .ops import bind, cleanup, superpose, unbind

__all__ = [
    "PhaseNuggetMemory",
    "RecallResult",
    "CRTValueCodebook",
    "crt_combine",
    "key_vector",
    "bind",
    "unbind",
    "superpose",
    "cleanup",
    "recall_chain",
    "compose_reach",
    "lifted_reach",
    "ChainResult",
]

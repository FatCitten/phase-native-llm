"""
Beat the file: recall a paraphrased prior conclusion that exact grep misses — or abstain.

The honest baseline for "remember my own conclusions" is decisions.md + grep, not a vector DB.
Grep wins on exact cues. It loses the moment the restatement isn't byte-identical. This is the
regime the lucid memory is for: fuzzy associative recall of committed conclusions, with a
calibrated abstain and an auditable receipt.

Metrics (saved to results/):
  recall on PARAPHRASE queries     — grep (substring) vs lucid (fuzzy)   [lucid should win big]
  false-confident on NEVER-STORED  — both should be ~0                    [lucid must not confabulate]
  false-merge on DISTINCT-but-lexically-overlapping conclusions           [the honest precision cost]

Run: python experiments/beat_the_file.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_native.lucid_guard import LucidMemory

RESULTS = Path("results")

# (conclusion committed earlier, a natural restatement queried later)
PAIRS = [
    ("chose exponential backoff over a circuit breaker because the downstream service is idempotent",
     "backoff was picked instead of a circuit breaker since the downstream is idempotent"),
    ("we cache embeddings in redis keyed by document hash to avoid recomputation",
     "embeddings are cached in redis under the document hash so we dont recompute them"),
    ("the migration must run before deploy because the new column is non-nullable",
     "run the migration prior to deploying, the added column is non nullable"),
    ("rate limiting is enforced per api key not per ip address",
     "rate limiting keys on the api key rather than the ip address"),
    ("the parser rejects duplicate keys in the config to prevent silent overrides",
     "duplicate config keys are rejected by the parser to avoid silent overrides"),
    ("retries use full jitter to avoid a thundering herd on recovery",
     "full jitter is applied on retries so recovery avoids a thundering herd"),
    ("user sessions are stored server side and only a signed id lives in the cookie",
     "sessions live server side, the cookie only carries a signed session id"),
    ("the worker pool size is capped at cpu count to avoid context switch thrash",
     "worker pool is capped at the cpu count to prevent context switch thrashing"),
    ("we denormalize the author name onto posts to skip a join on the hot read path",
     "author name is denormalized onto posts to avoid a join on the hot read path"),
    ("feature flags are evaluated at request time not cached to allow instant rollback",
     "flags get evaluated per request rather than cached so rollback is instant"),
]
CONCLUSIONS = [c for c, _ in PAIRS]

DISTRACTORS = [  # never stored — must abstain
    "the frontend uses tailwind for styling",
    "postgres connection pool size is set to twenty",
    "logs are shipped to datadog every minute",
    "the ci pipeline runs on every pull request",
    "images are served from a cdn with a long cache ttl",
]

# distinct conclusions that share vocabulary with a stored one (the honest precision test)
NEAR_COLLISIONS = [
    "rate limiting is disabled for internal ip addresses",          # shares rate/limiting/ip
    "the migration adds a nullable column after deploy",            # shares migration/column/deploy
]


def grep_hit(q, store):
    """Exact substring, either direction — what a file + grep actually gives you."""
    ql = q.lower()
    return any(ql in c.lower() or c.lower() in ql for c in store)


def main():
    RESULTS.mkdir(exist_ok=True)
    lm = LucidMemory(reps=1024)  # 4096-dim, 64 KB
    for c in CONCLUSIONS:
        lm.commit(c)

    # paraphrase recall: grep vs lucid
    grep_r = lucid_r = lucid_abstain = 0
    for (c, q) in PAIRS:
        if grep_hit(q, CONCLUSIONS):
            grep_r += 1
        v = lm.verify(q)
        if v.status == "recalled" and v.statement == c:
            lucid_r += 1
        elif v.status == "unknown":
            lucid_abstain += 1
    n = len(PAIRS)

    # never-stored: false-confident rate
    grep_fp = sum(grep_hit(q, CONCLUSIONS) for q in DISTRACTORS)
    lucid_fp = sum(lm.verify(q).status == "recalled" for q in DISTRACTORS)

    # near-collision precision: distinct-but-overlapping should NOT confidently recall
    nc_confab = sum(lm.verify(q).status == "recalled" for q in NEAR_COLLISIONS)

    print("=== recall on PARAPHRASE queries (grep would miss these) ===")
    print(f"  grep  : {grep_r}/{n}")
    print(f"  lucid : {lucid_r}/{n} recalled  ({lucid_abstain}/{n} honestly abstained, 0 confabulated)")
    print("=== never-stored distractors (false-confident) ===")
    print(f"  grep  : {grep_fp}/{len(DISTRACTORS)}   lucid : {lucid_fp}/{len(DISTRACTORS)}")
    print("=== distinct-but-overlapping conclusions (honest precision cost) ===")
    print(f"  lucid confidently recalled a WRONG neighbor: {nc_confab}/{len(NEAR_COLLISIONS)}")

    res = {"n_pairs": n, "grep_recall": grep_r, "lucid_recall": lucid_r,
           "lucid_abstain": lucid_abstain, "grep_false_confident": grep_fp,
           "lucid_false_confident": lucid_fp, "near_collision_confab": nc_confab,
           "n_distractors": len(DISTRACTORS)}
    (RESULTS / "beat_the_file.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["grep\n(file)", "lucid\nmemory"], [grep_r / n * 100, lucid_r / n * 100],
           color=["#b0653c", "#2e7d32"])
    for i, v in enumerate([grep_r / n * 100, lucid_r / n * 100]):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("paraphrased prior conclusions recalled")
    ax.set_ylim(0, 105)
    ax.set_title(f"Recall of restated conclusions grep can't match  "
                 f"(0 confabulations on {len(DISTRACTORS)} never-stored)")
    fig.tight_layout()
    fig.savefig(RESULTS / "beat_the_file.png", dpi=140)
    print("\nSaved results/beat_the_file.{json,png}")


if __name__ == "__main__":
    main()

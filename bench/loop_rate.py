#!/usr/bin/env python3
"""Degenerate-repetition (looping) rate for two endpoints, head to head.

Quantization can damage the low-margin policy decisions that keep a model from
repeating itself, so a re-quant has to be judged on loop rate, not just tok/s.
This drives long, loop-prone generations at both endpoints with identical
prompts/seeds and reports how often output degenerates.

Detection (no model, no judge — purely structural):
  * verbatim block cycling: the longest suffix-repeat of a normalized line
    n-gram, i.e. the same 2-line block emitted >= REPEAT_MIN times in a row
  * global line duplication: unique_lines / total_lines below a floor
  * trigram entropy collapse: distinct-3gram / total-3gram below a floor
  * truncation at max_tokens with an unterminated sentence (soft signal)

    python3 bench/loop_rate.py --a http://HOST_A/v1 --model-a NAME \
                               --b http://HOST_B/v1 --model-b NAME --runs 6
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.request

REPEAT_MIN = 3          # a block seen this many times back-to-back = cycling
UNIQ_LINE_FLOOR = 0.55  # unique/total lines below this = heavy duplication
TRIGRAM_FLOOR = 0.30    # distinct/total word-trigrams below this = collapse

# Two profiles.
#
# "agentic" reproduces the shape that actually loops in production. Captured
# stoploop cuts (17 of them) share a signature: the model is mid-task inside an
# AGENT HARNESS, a tool is failing or unavailable, and the REASONING channel
# cycles a ~400-2600 char block of "let me try X / actually X won't work / let
# me try a different approach" ~3 times. It is a stuck-tool deliberation loop,
# not a text-degeneration loop, so ordinary long-form prompts never surface it.
AGENTIC_SYSTEM = (
    "You are an autonomous coding agent inside a harness with these tools: "
    "shell(cmd), browser_navigate(url), browser_screenshot(), read_file(path), "
    "write_file(path, content). Think step by step before acting. You cannot "
    "set environment variables for the browser tool, and the browser runs in a "
    "different environment than the shell."
)
AGENTIC_PROMPTS = [
    "Take a screenshot of the dashboard running on http://127.0.0.1:8090. The "
    "browser tool returns 'blocked: private URL' and needs ALLOW_PRIVATE_URLS=1, "
    "which you cannot set. Keep working the problem until you have the "
    "screenshot.",
    "The test suite fails with 'ModuleNotFoundError: no module named app'. "
    "pip install fails: read-only filesystem. The venv you need is at a path "
    "you cannot read. Get the tests running.",
    "Deploy the service. `docker` is not on PATH, `sudo` prompts for a password "
    "you do not have, and the systemd unit references a file that does not "
    "exist. Complete the deployment.",
    "Find why the app returns stale reads. The logs directory is empty, the "
    "metrics endpoint 404s, and you cannot restart the service. Diagnose it.",
    "Convert report.xlsx to CSV. pandas is not installed, the network is "
    "unavailable, and the file is binary. Produce the CSV.",
    "Fix the failing lint job. The linter config is generated at build time by "
    "a script you cannot run, and the CI log is truncated. Make it pass.",
]

# Prompts chosen to invite looping: open-ended enumeration, self-reference,
# long structured output, and an under-specified agentic task.
PROMPTS = [
    "List 40 distinct edge cases for a URL parser. Number them and give one "
    "sentence each. Do not repeat yourself.",
    "Write a detailed step-by-step plan to refactor a 200k-line monolith into "
    "services. Cover discovery, seams, data, rollout and rollback.",
    "Enumerate every HTTP status code you know with a one-line description, "
    "then explain when each is misused in practice.",
    "Think through, at length, how you would debug a distributed system where "
    "one node intermittently returns stale reads. Consider every hypothesis.",
    "Write a complete Python implementation of a B-tree with insert, delete, "
    "search, and rebalancing, plus docstrings and tests.",
    "Describe the design of a rate limiter, then critique your own design, "
    "then revise it, then critique the revision.",
]


def generate(
    url: str, model: str, prompt: str, max_tokens: int, seed: int,
    system: str | None = None,
) -> tuple[str, str]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "seed": seed,
            "messages": msgs,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", body, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning") or "")
    return text, ch.get("finish_reason") or ""


def norm_lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        s = re.sub(r"\s+", " ", ln).strip().lower()
        s = re.sub(r"^\W*\d+[\.\)]\s*", "", s)  # drop list numbering
        if s:
            out.append(s)
    return out


def max_block_cycle(lines: list[str], block: int = 2) -> int:
    """Largest number of back-to-back repeats of any `block`-line window."""
    if len(lines) < block * 2:
        return 1
    best = 1
    i = 0
    while i + block <= len(lines):
        w = lines[i : i + block]
        reps = 1
        j = i + block
        while j + block <= len(lines) and lines[j : j + block] == w:
            reps += 1
            j += block
        best = max(best, reps)
        i += 1
    return best


def trigram_ratio(text: str) -> float:
    w = re.findall(r"\w+", text.lower())
    if len(w) < 30:
        return 1.0
    tris = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
    return len(set(tris)) / len(tris)


def analyze(text: str, finish: str, max_tokens: int) -> dict:
    lines = norm_lines(text)
    uniq = len(set(lines)) / len(lines) if lines else 1.0
    cycle = max_block_cycle(lines)
    tri = trigram_ratio(text)
    looped = cycle >= REPEAT_MIN or uniq < UNIQ_LINE_FLOOR or tri < TRIGRAM_FLOOR
    return {
        "looped": looped,
        "max_block_repeats": cycle,
        "uniq_line_ratio": round(uniq, 3),
        "trigram_ratio": round(tri, 3),
        "truncated": finish == "length",
        "chars": len(text),
    }


def run_side(
    label: str, url: str, model: str, runs: int, max_tokens: int,
    profile: str = "generic",
) -> dict:
    prompts = AGENTIC_PROMPTS if profile == "agentic" else PROMPTS
    system = AGENTIC_SYSTEM if profile == "agentic" else None
    rows = []
    for i in range(runs):
        prompt = prompts[i % len(prompts)]
        try:
            text, finish = generate(
                url, model, prompt, max_tokens, seed=1000 + i, system=system
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {label} run{i + 1}: ERROR {e}", flush=True)
            continue
        a = analyze(text, finish, max_tokens)
        rows.append(a)
        flag = "LOOP" if a["looped"] else "ok  "
        print(
            f"  {label} run{i + 1}: {flag} rep={a['max_block_repeats']} "
            f"uniq={a['uniq_line_ratio']} tri={a['trigram_ratio']} "
            f"chars={a['chars']}{' trunc' if a['truncated'] else ''}",
            flush=True,
        )
    n = len(rows)
    looped = sum(1 for r in rows if r["looped"])
    return {
        "label": label,
        "runs": n,
        "looped": looped,
        "loop_rate": round(looped / n, 3) if n else None,
        "median_uniq": statistics.median([r["uniq_line_ratio"] for r in rows]) if rows else None,
        "median_trigram": statistics.median([r["trigram_ratio"] for r in rows]) if rows else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="endpoint A base url (/v1)")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--b", help="endpoint B base url (/v1)")
    ap.add_argument("--model-b")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--profile",
        choices=("generic", "agentic"),
        default="agentic",
        help="agentic reproduces the stuck-tool reasoning loop seen in production",
    )
    args = ap.parse_args()

    out = {}
    print(f"A = {args.model_a}", flush=True)
    out["A"] = run_side("A", args.a, args.model_a, args.runs, args.max_tokens, args.profile)
    if args.b:
        print(f"B = {args.model_b}", flush=True)
        out["B"] = run_side(
            "B", args.b, args.model_b or args.model_a, args.runs, args.max_tokens,
            args.profile,
        )

    print()
    for k, v in out.items():
        print(
            f"{k} ({v['label']}): loop rate {v['looped']}/{v['runs']} "
            f"= {v['loop_rate']}  median uniq-lines {v['median_uniq']}  "
            f"median trigram {v['median_trigram']}",
            flush=True,
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

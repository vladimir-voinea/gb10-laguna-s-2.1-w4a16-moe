#!/usr/bin/env python3
"""End-to-end single-stream decode TPS against an OpenAI-compatible endpoint.

Streams a completion and reports decode tok/s measured between the first and
last received token (prefill excluded). stdlib only.

    python3 bench/e2e_tps.py --url http://HOST:PORT/v1 --model NAME \
        --runs 2 --max-tokens 800
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

CODING = (
    "Write a complete Python module implementing an LRU cache with TTL "
    "support, generics, full type hints, docstrings and a small test suite. "
    "Do not stop until the module is complete."
)
PROSE = (
    "Write a detailed essay about the history of container shipping and its "
    "effect on global trade, with concrete examples and numbers."
)


def one_run(url: str, model: str, prompt: str, max_tokens: int) -> tuple[int, float]:
    body = json.dumps(
        {
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        body,
        {"content-type": "application/json"},
    )
    n = 0
    t_first = t_last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                d = json.loads(line[6:])["choices"][0].get("delta") or {}
            except Exception:
                continue
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
    if n < 2 or t_first is None or t_last is None or t_last <= t_first:
        raise RuntimeError(f"too few tokens streamed (n={n})")
    return n, (n - 1) / (t_last - t_first)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    for label, prompt in (("coding", CODING), ("prose", PROSE)):
        rates = []
        for i in range(args.runs):
            n, tps = one_run(args.url, args.model, prompt, args.max_tokens)
            rates.append(tps)
            print(f"  {label} run{i + 1}: {n} chunks  {tps:.1f} tok/s", flush=True)
        print(
            f"{label}: median {statistics.median(rates):.1f} tok/s "
            f"(runs: {', '.join(f'{r:.1f}' for r in rates)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

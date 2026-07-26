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
            # Chunks ≠ tokens: speculative decoding emits several tokens per
            # SSE event, so rate must come from the server's own token count.
            "stream_options": {"include_usage": True},
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
    n_chunks = 0
    completion_tokens = None
    t_first = t_last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                obj = json.loads(line[6:])
            except Exception:
                continue
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])
            choices = obj.get("choices") or []
            d = (choices[0].get("delta") or {}) if choices else {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
    if n_chunks < 2 or t_first is None or t_last is None or t_last <= t_first:
        raise RuntimeError(f"too few chunks streamed (n={n_chunks})")
    n_tokens = completion_tokens if completion_tokens else n_chunks
    # Tokens delivered after the first chunk, over the first→last chunk window.
    per_chunk = n_tokens / n_chunks
    return n_tokens, (n_tokens - per_chunk) / (t_last - t_first)


FILLER = (
    "The container revolutionized logistics by standardizing the unit of "
    "cargo, collapsing port dwell times and slashing theft and breakage. "
)


def long_prompt(base: str, approx_tokens: int) -> str:
    """Pad a prompt to roughly approx_tokens (chars/4 heuristic) to exercise
    long-context prefill and decode-at-depth."""
    if approx_tokens <= 0:
        return base
    pad = FILLER * max(1, (approx_tokens * 4 - len(base)) // len(FILLER))
    return (
        "Background material follows; read it, then answer the final request."
        "\n\n" + pad + "\n\n" + base
    )


def concurrent_runs(
    url: str, model: str, prompt: str, max_tokens: int, conc: int
) -> None:
    """conc parallel streams; per-stream decode tok/s + wall aggregate."""
    import threading

    results: list[tuple[int, float] | Exception] = [None] * conc  # type: ignore[list-item]

    def worker(i: int) -> None:
        try:
            # Vary the prompt per stream so prefix caching cannot serialize them.
            results[i] = one_run(url, model, f"[stream {i}] {prompt}", max_tokens)
        except Exception as e:  # noqa: BLE001 - report, don't kill the matrix
            results[i] = e

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_tokens = 0
    per_stream = []
    for r in results:
        if isinstance(r, Exception):
            print(f"    stream error: {r}", flush=True)
            continue
        n, tps = r
        total_tokens += n
        per_stream.append(tps)
    if per_stream:
        print(
            f"  c{conc}: aggregate {total_tokens / wall:.1f} tok/s  "
            f"per-stream median {statistics.median(per_stream):.1f} tok/s  "
            f"({len(per_stream)}/{conc} streams ok)",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument(
        "--concurrency",
        type=str,
        default="1",
        help="comma list of stream counts, e.g. 1,4,8",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="pad prompts to ~N tokens to test long-context prefill + decode",
    )
    args = ap.parse_args()

    conc_list = [int(c) for c in args.concurrency.split(",") if c.strip()]
    for label, prompt in (("coding", CODING), ("prose", PROSE)):
        prompt = long_prompt(prompt, args.prompt_tokens)
        for conc in conc_list:
            if conc <= 1:
                rates = []
                for i in range(args.runs):
                    n, tps = one_run(args.url, args.model, prompt, args.max_tokens)
                    rates.append(tps)
                    print(
                        f"  {label} run{i + 1}: {n} chunks  {tps:.1f} tok/s",
                        flush=True,
                    )
                print(
                    f"{label} c1: median {statistics.median(rates):.1f} tok/s "
                    f"(runs: {', '.join(f'{r:.1f}' for r in rates)})",
                    flush=True,
                )
            else:
                print(f"{label} c{conc}:", flush=True)
                concurrent_runs(args.url, args.model, prompt, args.max_tokens, conc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

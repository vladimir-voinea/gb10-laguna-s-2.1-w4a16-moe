#!/usr/bin/env python3
"""Tune Triton/CUDA fused-MoE configs for int4_w4a16 on the local GPU.

Produces a vLLM-compatible config JSON under ``configs/`` whose top-level keys
are M-bucket strings and whose values are kernel config dicts
(``BLOCK_SIZE_{M,N,K}``, ``GROUP_SIZE_M``, ``num_warps``, ``num_stages``, …).

Candidates are injected by monkeypatching
``vllm…fused_moe.get_moe_configs`` / ``try_get_optimal_moe_config`` — installed
vLLM sources are never edited. Weight construction, routing, and CUDA-event
timing reuse ``bench_moe``.

Run inside a vLLM container with the repo bind-mounted, e.g.::

    python3 /repo/bench/tune_configs.py --out-dir /repo/configs
    python3 /repo/bench/tune_configs.py --quick --M 1,16,256
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

# Reuse bench machinery (same directory when run as a script; package-safe too).
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import bench_moe as bm  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_M_LIST = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)
DTYPE_STR = "int4_w4a16"
# Decode-band targets from TASK-02 (acceptance).
DECODE_M = frozenset({1, 4, 16, 64, 256})
TARGET_GBS = 170.0
# CUDA moe_wna16 requires BLOCK_SIZE_K // group_size ∈ {1, 2, 4, 8}.
CUDA_K_OVER_GROUP = frozenset({1, 2, 4, 8})


# ---------------------------------------------------------------------------
# Config constraints / path selection (mirrors reference/fused_moe.py)
# ---------------------------------------------------------------------------


def should_moe_wna16_use_cuda(
    M: int,
    topk: int,
    E: int,
    group_size: int,
    bit: int = 4,
) -> bool:
    """Same predicate as ``fused_moe.should_moe_wna16_use_cuda`` (CUDA assumed)."""
    num_valid_tokens = M * topk
    return (
        bit == 4
        and group_size in (32, 64, 128)
        and num_valid_tokens / max(E, 1) <= 6
    )


def cuda_block_k_ok(block_k: int, group_size: int) -> bool:
    if group_size <= 0 or block_k % group_size != 0:
        return False
    return (block_k // group_size) in CUDA_K_OVER_GROUP


def block_k_divides_dims(block_k: int, K: int, N: int) -> bool:
    """BLOCK_SIZE_K is shared across both MoE GEMMs (K_in = K and K_in = N)."""
    return K % block_k == 0 and N % block_k == 0


def config_is_statically_valid(
    cfg: dict[str, int],
    *,
    M: int,
    K: int,
    N: int,
    E: int,
    topk: int,
    group_size: int,
) -> tuple[bool, str]:
    """Reject configs that cannot run (CUDA ratio / divisibility) without a kernel launch."""
    bm_ = cfg.get("BLOCK_SIZE_M")
    bn = cfg.get("BLOCK_SIZE_N")
    bk = cfg.get("BLOCK_SIZE_K")
    if bm_ is None or bn is None or bk is None:
        return False, "missing BLOCK_SIZE_*"
    if bm_ < 1 or bn < 1 or bk < 1:
        return False, "non-positive block size"
    if not block_k_divides_dims(bk, K, N):
        return False, f"BLOCK_SIZE_K={bk} does not divide K={K} and N={N}"
    # Always enforce the CUDA ratio: small-M buckets take the CUDA path, and
    # keeping K legal everywhere is cheap and crash-safe if the threshold moves.
    if not cuda_block_k_ok(bk, group_size):
        return (
            False,
            f"BLOCK_SIZE_K//group={bk // group_size} not in {sorted(CUDA_K_OVER_GROUP)}",
        )
    _ = (M, E, topk)  # kept for call-site symmetry / future soft filters
    return True, ""


def candidate_configs(
    M: int,
    *,
    group_size: int,
    K: int,
    N: int,
    E: int,
    topk: int,
    quick: bool,
) -> list[dict[str, int]]:
    """Build a sensible search grid for one M bucket."""
    use_cuda = should_moe_wna16_use_cuda(M, topk, E, group_size)

    if quick:
        block_m_opts = [16, 32, 64]
        block_n_opts = [32, 64, 128, 256]
        block_k_opts = [32, 64, 128, 256]
        group_m_opts = [1, 8]
        warps_opts = [4, 8]
        stages_opts = [2, 3, 4]
    else:
        block_m_opts = [16, 32, 64, 128]
        block_n_opts = [16, 32, 64, 128, 256, 512]
        block_k_opts = [32, 64, 128, 256]
        group_m_opts = [1, 4, 8, 16, 32]
        warps_opts = [2, 4, 8]
        stages_opts = [2, 3, 4, 5]

    # Prefer tiles that fit the batch: CUDA default uses min(16, M).
    if M <= 16:
        block_m_opts = sorted(set([min(16, max(1, M)), 16, 32] + (
            [64] if not quick else []
        )))
    elif M <= 64:
        block_m_opts = [x for x in block_m_opts if x <= 64] or block_m_opts
    # Large M: drop tiny M tiles (too many blocks).
    if M >= 512:
        block_m_opts = [x for x in block_m_opts if x >= 32] or block_m_opts
        block_n_opts = [x for x in block_n_opts if x >= 32] or block_n_opts

    # CUDA path ignores warps/stages; keep a single value to shrink the grid.
    if use_cuda:
        warps_opts = [4]
        stages_opts = [3]
        group_m_opts = [1]

    out: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for bm_, bn, bk, gm, nw, ns in itertools.product(
        block_m_opts,
        block_n_opts,
        block_k_opts,
        group_m_opts,
        warps_opts,
        stages_opts,
    ):
        cfg = {
            "BLOCK_SIZE_M": int(bm_),
            "BLOCK_SIZE_N": int(bn),
            "BLOCK_SIZE_K": int(bk),
            "GROUP_SIZE_M": int(gm),
            "SPLIT_K": 1,
            "num_warps": int(nw),
            "num_stages": int(ns),
        }
        ok, _ = config_is_statically_valid(
            cfg, M=M, K=K, N=N, E=E, topk=topk, group_size=group_size
        )
        if not ok:
            continue
        key = (
            cfg["BLOCK_SIZE_M"],
            cfg["BLOCK_SIZE_N"],
            cfg["BLOCK_SIZE_K"],
            cfg["GROUP_SIZE_M"],
            cfg["num_warps"],
            cfg["num_stages"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Monkeypatch injection
# ---------------------------------------------------------------------------


@contextmanager
def inject_moe_config(config: dict[str, int], M: int) -> Iterator[None]:
    """Force ``try_get_optimal_moe_config`` / ``get_moe_configs`` to return ``config``.

    Patches the live vLLM module so installed sources stay untouched.
    """
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod

    cfg_copy = dict(config)
    # Map with a single M key → nearest-key lookup always returns our candidate.
    forced_map: dict[int, dict[str, int]] = {int(M): dict(cfg_copy)}

    def fake_get_moe_configs(
        E: int,
        N: int,
        dtype: str | None,
        block_n: int | None = None,
        block_k: int | None = None,
    ) -> dict[int, Any]:
        _ = (E, N, dtype, block_n, block_k)
        return forced_map

    def fake_try_get_optimal_moe_config(
        w1_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        top_k: int,
        dtype: str | None,
        M_arg: int,
        block_shape: list[int] | None = None,
    ) -> dict[str, int]:
        _ = (w1_shape, w2_shape, top_k, dtype, M_arg, block_shape)
        return dict(cfg_copy)

    orig_get = fused_moe_mod.get_moe_configs
    orig_try = fused_moe_mod.try_get_optimal_moe_config

    # Drop LRU cache on the real get_moe_configs if present.
    if hasattr(orig_get, "cache_clear"):
        try:
            orig_get.cache_clear()
        except Exception:  # noqa: BLE001
            pass

    fused_moe_mod.get_moe_configs = fake_get_moe_configs  # type: ignore[assignment]
    fused_moe_mod.try_get_optimal_moe_config = fake_try_get_optimal_moe_config  # type: ignore[assignment]
    try:
        yield
    finally:
        fused_moe_mod.get_moe_configs = orig_get  # type: ignore[assignment]
        fused_moe_mod.try_get_optimal_moe_config = orig_try  # type: ignore[assignment]
        if hasattr(orig_get, "cache_clear"):
            try:
                orig_get.cache_clear()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Measure one candidate
# ---------------------------------------------------------------------------


def measure_candidate(
    config: dict[str, int],
    *,
    M: int,
    hidden: torch.Tensor,
    weights: bm.MoEWeights,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    nbytes: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Time one config; return status + bandwidth. Never raises."""
    try:
        with inject_moe_config(config, M):
            # One dry run: catches compile / CUDA-constraint errors.
            out = bm.run_triton(hidden, weights, topk_w, topk_ids)
            torch.cuda.synchronize()
            _ = out

            def _call() -> torch.Tensor:
                return bm.run_triton(hidden, weights, topk_w, topk_ids)

            times = bm.cuda_time_ms(_call, warmup, iters)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "invalid",
            "reason": f"{type(exc).__name__}: {exc}",
            "config": config,
        }

    med = bm.p50(times)
    gbs = (nbytes / (med * 1e-3)) / 1e9 if med > 0 else float("nan")
    return {
        "status": "ok",
        "config": config,
        "p50_ms": med,
        "GB_s": gbs,
        "times_ms": times,
    }


def correctness_check(
    config: dict[str, int],
    *,
    weights: bm.MoEWeights,
    device: torch.device,
    dtype: torch.dtype,
    E: int,
    K: int,
    topk: int,
    seed: int,
) -> dict[str, Any]:
    """Spot-check config at M=16 against the PyTorch reference."""
    M = bm.CORRECTNESS_M
    h = bm.make_hidden(M, K, device, dtype, seed)
    tw, ti = bm.make_routing(M, E, topk, device, dtype, seed)
    ref = bm.run_reference(h, weights, tw, ti)
    torch.cuda.synchronize()
    try:
        with inject_moe_config(config, M):
            out = bm.run_triton(h, weights, tw, ti)
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {
            "pass": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "max_abs": None,
            "rel_frob": None,
        }
    ma = bm.max_abs_err(out, ref)
    rf = bm.rel_frobenius(out, ref)
    return {
        "pass": rf <= bm.REL_FROB_RTOL,
        "max_abs": ma,
        "rel_frob": rf,
        "rtol": bm.REL_FROB_RTOL,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def sanitize_device_name(name: str) -> str:
    return name.replace(" ", "_")


def config_filename(E: int, N: int, device: str, dtype: str = DTYPE_STR) -> str:
    return f"E={E},N={N},device_name={sanitize_device_name(device)},dtype={dtype}.json"


def write_notes(
    path: Path | str,
    *,
    device: str,
    shapes: dict[str, Any],
    roofline: float,
    winners: dict[int, dict[str, Any]],
    target_gbs: float,
) -> None:
    lines: list[str] = [
        "# GB10 int4_w4a16 fused-MoE config tune notes",
        "",
        f"- device: `{device}`",
        f"- shapes: E={shapes['E']} K={shapes['K']} N={shapes['N']} "
        f"topk={shapes['topk']} group={shapes['group']}",
        f"- roofline: {roofline:g} GB/s",
        f"- decode-band target: p50 ≥ {target_gbs:g} GB/s for M ∈ "
        f"{{{', '.join(str(m) for m in sorted(DECODE_M))}}}",
        "",
        "## Winning table",
        "",
        "| M | BLOCK_SIZE_M/N/K | GROUP_SIZE_M | warps | stages | GB/s | %roofline | path |",
        "|---|------------------|--------------|-------|--------|------|-----------|------|",
    ]
    decode_miss: list[str] = []
    for M in sorted(winners):
        w = winners[M]
        if w.get("status") != "ok":
            lines.append(
                f"| {M} | — | — | — | — | FAIL | — | {w.get('reason', 'invalid')} |"
            )
            continue
        c = w["config"]
        gbs = float(w["GB_s"])
        pct = 100.0 * gbs / roofline if roofline > 0 else float("nan")
        path = "cuda" if w.get("use_cuda") else "triton"
        lines.append(
            f"| {M} | {c['BLOCK_SIZE_M']}/{c['BLOCK_SIZE_N']}/{c['BLOCK_SIZE_K']} "
            f"| {c['GROUP_SIZE_M']} | {c['num_warps']} | {c['num_stages']} "
            f"| {gbs:.1f} | {pct:.1f} | {path} |"
        )
        if M in DECODE_M and gbs < target_gbs:
            decode_miss.append(f"M={M}: {gbs:.1f} GB/s")

    lines.extend(["", "## Acceptance"])
    if not decode_miss:
        lines.append(
            f"Decode-band target met (≥ {target_gbs:g} GB/s) for all "
            f"M ∈ {sorted(DECODE_M)}."
        )
    else:
        ceil = min(
            float(winners[m]["GB_s"])
            for m in DECODE_M
            if m in winners and winners[m].get("status") == "ok"
        ) if any(
            m in winners and winners[m].get("status") == "ok" for m in DECODE_M
        ) else 0.0
        # Report measured ceiling (best decode p50) as justification for TASK-03.
        best_decode = max(
            (
                float(winners[m]["GB_s"])
                for m in DECODE_M
                if m in winners and winners[m].get("status") == "ok"
            ),
            default=0.0,
        )
        worst_decode = min(
            (
                float(winners[m]["GB_s"])
                for m in DECODE_M
                if m in winners and winners[m].get("status") == "ok"
            ),
            default=0.0,
        )
        lines.append(
            f"Decode-band target **not** fully met (target {target_gbs:g} GB/s)."
        )
        lines.append(
            f"Measured decode ceiling by config choice alone: "
            f"best={best_decode:.1f} GB/s, worst-decode-bucket={worst_decode:.1f} GB/s "
            f"({100.0 * best_decode / roofline:.1f}% / "
            f"{100.0 * worst_decode / roofline:.1f}% of roofline)."
        )
        lines.append("Misses: " + "; ".join(decode_miss) + ".")
        lines.append(
            "This measured ceiling is the justification for a custom kernel "
            "(TASK-03) if config tuning alone cannot close the gap."
        )
        _ = ceil

    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_m_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tune int4_w4a16 fused-MoE configs for the local GPU"
    )
    p.add_argument("--E", type=int, default=bm.DEFAULT_E)
    p.add_argument("--K", type=int, default=bm.DEFAULT_K)
    p.add_argument("--N", type=int, default=bm.DEFAULT_N)
    p.add_argument("--topk", type=int, default=bm.DEFAULT_TOPK)
    p.add_argument("--group", type=int, default=bm.DEFAULT_GROUP)
    p.add_argument(
        "--M",
        type=str,
        default=",".join(str(m) for m in DEFAULT_M_LIST),
        help="Comma-separated M buckets to tune",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roofline", type=float, default=bm.DEFAULT_ROOFLINE_GBS)
    p.add_argument("--target-gbs", type=float, default=TARGET_GBS)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarser candidate grid and fewer timing iters",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs"),
        help="Directory for the config JSON and NOTES-gb10.md",
    )
    p.add_argument(
        "--device-name",
        type=str,
        default=None,
        help="Override device name used in the config filename "
        "(default: torch CUDA device name)",
    )
    p.add_argument(
        "--log-invalid",
        action="store_true",
        help="Print every invalid candidate (noisy)",
    )
    args = p.parse_args(argv)

    if args.quick:
        if args.warmup == 5:
            args.warmup = 2
        if args.iters == 20:
            args.iters = 5

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    dtype = torch.bfloat16
    E, K, N, topk, group = args.E, args.K, args.N, args.topk, args.group
    m_list = parse_m_list(args.M)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_name = args.device_name or bm.device_name()
    print(
        f"device={dev_name}  E={E} K={K} N={N} topk={topk} group={group} "
        f"dtype={DTYPE_STR}  quick={args.quick}  M={m_list}",
        flush=True,
    )
    print("building synthetic W4A16 weights …", flush=True)
    t0 = time.time()
    weights = bm.make_weights(E, K, N, group, dtype, device, args.seed)
    torch.cuda.synchronize()
    print(f"weights ready in {time.time() - t0:.1f}s", flush=True)

    winners: dict[int, dict[str, Any]] = {}
    invalid_counts: dict[int, int] = {}
    trial_counts: dict[int, int] = {}

    for M in m_list:
        use_cuda = should_moe_wna16_use_cuda(M, topk, E, group)
        cands = candidate_configs(
            M,
            group_size=group,
            K=K,
            N=N,
            E=E,
            topk=topk,
            quick=args.quick,
        )
        trial_counts[M] = len(cands)
        print(
            f"\n=== M={M}  path={'cuda' if use_cuda else 'triton'}  "
            f"candidates={len(cands)} ===",
            flush=True,
        )
        if not cands:
            print(f"M={M}: no candidates after static filter", flush=True)
            winners[M] = {"status": "invalid", "reason": "empty candidate set"}
            continue

        hidden = bm.make_hidden(M, K, device, dtype, args.seed)
        topk_w, topk_ids = bm.make_routing(M, E, topk, device, dtype, args.seed)
        n_act = bm.unique_expert_count(topk_ids)
        nbytes = bm.weight_bytes_for_experts(n_act, K, N, group)

        best: dict[str, Any] | None = None
        n_invalid = 0
        for i, cfg in enumerate(cands):
            res = measure_candidate(
                cfg,
                M=M,
                hidden=hidden,
                weights=weights,
                topk_w=topk_w,
                topk_ids=topk_ids,
                nbytes=nbytes,
                warmup=args.warmup,
                iters=args.iters,
            )
            if res["status"] != "ok":
                n_invalid += 1
                if args.log_invalid:
                    print(
                        f"  [{i + 1}/{len(cands)}] INVALID "
                        f"{cfg}  {res.get('reason')}",
                        flush=True,
                    )
                continue
            gbs = float(res["GB_s"])
            tag = ""
            if best is None or gbs > float(best["GB_s"]):
                best = res
                tag = " *best*"
            if (i + 1) % max(1, len(cands) // 10) == 0 or tag:
                print(
                    f"  [{i + 1}/{len(cands)}] "
                    f"M{cfg['BLOCK_SIZE_M']}/N{cfg['BLOCK_SIZE_N']}/"
                    f"K{cfg['BLOCK_SIZE_K']} g{cfg['GROUP_SIZE_M']} "
                    f"w{cfg['num_warps']} s{cfg['num_stages']}  "
                    f"{gbs:.1f} GB/s  p50={res['p50_ms']:.3f} ms{tag}",
                    flush=True,
                )

        invalid_counts[M] = n_invalid
        if best is None:
            print(f"M={M}: all {len(cands)} candidates invalid", flush=True)
            winners[M] = {
                "status": "invalid",
                "reason": f"all {len(cands)} candidates failed",
                "invalid": n_invalid,
            }
            continue

        best["use_cuda"] = use_cuda
        best["unique_experts"] = n_act
        best["weight_bytes"] = nbytes
        winners[M] = best
        print(
            f"M={M} WINNER  {best['GB_s']:.1f} GB/s  "
            f"p50={best['p50_ms']:.3f} ms  "
            f"cfg={best['config']}  "
            f"(invalid={n_invalid}/{len(cands)})",
            flush=True,
        )

    # Correctness spot-check: prefer the M=16 winner; else any ok winner.
    corr_cfg: dict[str, int] | None = None
    if 16 in winners and winners[16].get("status") == "ok":
        corr_cfg = winners[16]["config"]
    else:
        for M in sorted(winners):
            if winners[M].get("status") == "ok":
                corr_cfg = winners[M]["config"]
                break

    corr: dict[str, Any] = {"pass": False, "reason": "no valid winner"}
    if corr_cfg is not None:
        print(
            f"\ncorrectness spot-check at M={bm.CORRECTNESS_M} "
            f"with cfg={corr_cfg}",
            flush=True,
        )
        corr = correctness_check(
            corr_cfg,
            weights=weights,
            device=device,
            dtype=dtype,
            E=E,
            K=K,
            topk=topk,
            seed=args.seed,
        )
        print(
            f"correctness: pass={corr.get('pass')}  "
            f"max_abs={corr.get('max_abs')}  rel_frob={corr.get('rel_frob')}  "
            f"reason={corr.get('reason')}",
            flush=True,
        )
        if not corr.get("pass"):
            print(
                "WARNING: correctness check failed; JSON still written for inspection",
                file=sys.stderr,
                flush=True,
            )

    # Emit vLLM config JSON (string keys, config dicts only).
    json_body: dict[str, Any] = {}
    try:
        import triton  # type: ignore

        json_body["triton_version"] = getattr(triton, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass

    for M in sorted(winners):
        w = winners[M]
        if w.get("status") == "ok":
            json_body[str(M)] = dict(w["config"])

    fname = config_filename(E, N, dev_name, DTYPE_STR)
    json_path = out_dir / fname
    json_path.write_text(json.dumps(json_body, indent=4) + "\n")
    print(f"\nwrote {json_path}", flush=True)

    notes_path = out_dir / "NOTES-gb10.md"
    write_notes(
        notes_path,
        device=dev_name,
        shapes={"E": E, "K": K, "N": N, "topk": topk, "group": group},
        roofline=args.roofline,
        winners=winners,
        target_gbs=args.target_gbs,
    )
    print(f"wrote {notes_path}", flush=True)

    # Sidecar metrics (not consumed by vLLM) for operator inspection.
    metrics_path = out_dir / (fname.replace(".json", ".metrics.json"))
    metrics = {
        "device": dev_name,
        "shapes": {
            "E": E,
            "K": K,
            "N": N,
            "topk": topk,
            "group": group,
            "dtype": DTYPE_STR,
        },
        "roofline_GB_s": args.roofline,
        "target_GB_s": args.target_gbs,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "quick": args.quick,
        "correctness": corr,
        "winners": {
            str(M): {
                k: v
                for k, v in w.items()
                if k != "times_ms"  # drop bulky arrays
            }
            for M, w in winners.items()
        },
        "invalid_counts": {str(k): v for k, v in invalid_counts.items()},
        "trial_counts": {str(k): v for k, v in trial_counts.items()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {metrics_path}", flush=True)

    # Summary table
    print(
        f"\n{'M':>6} {'GB/s':>10} {'%roof':>8} {'path':>6}  config",
        flush=True,
    )
    any_ok = False
    for M in sorted(winners):
        w = winners[M]
        if w.get("status") != "ok":
            print(f"{M:>6} {'FAIL':>10} {'—':>8} {'—':>6}  {w.get('reason')}", flush=True)
            continue
        any_ok = True
        gbs = float(w["GB_s"])
        pct = 100.0 * gbs / args.roofline if args.roofline > 0 else float("nan")
        path = "cuda" if w.get("use_cuda") else "triton"
        c = w["config"]
        print(
            f"{M:>6} {gbs:>10.1f} {pct:>8.1f} {path:>6}  "
            f"M{c['BLOCK_SIZE_M']}/N{c['BLOCK_SIZE_N']}/K{c['BLOCK_SIZE_K']} "
            f"g{c['GROUP_SIZE_M']} w{c['num_warps']} s{c['num_stages']}",
            flush=True,
        )

    if not any_ok:
        print("ERROR: no valid configs found for any M", file=sys.stderr)
        return 1
    if corr_cfg is not None and not corr.get("pass"):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)

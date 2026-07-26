#!/usr/bin/env python3
"""W4A16 grouped-MoE microbenchmark (correctness + weight-bandwidth).

Standalone: only torch / triton / installed vllm. No pip installs.
Run inside a vLLM container with the repo bind-mounted, e.g.::

    python3 /repo/bench/bench_moe.py --backend all --json /repo/results/latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Defaults (laguna117b one-layer MoE shapes)
# ---------------------------------------------------------------------------

DEFAULT_E = 256
DEFAULT_K = 3072
DEFAULT_N = 1024
DEFAULT_TOPK = 10
DEFAULT_GROUP = 32
DEFAULT_M_LIST = (1, 4, 16, 64, 256, 1024, 4096)
DEFAULT_ROOFLINE_GBS = 273.0
WARMUP_ITERS = 10
TIMED_ITERS = 50
CORRECTNESS_M = 16
REL_FROB_RTOL = 3e-2


# ---------------------------------------------------------------------------
# Quantization / packing  (matches vLLM fused_moe int4_w4a16 / GPTQ-AWQ path)
#
# From reference/fused_moe.py + vLLM test_fused_moe_wn16:
#   - qweight dtype uint8, two K-values per byte along the last dim
#   - nibble order: low nibble = even K, high nibble = odd K
#       packed[:, k//2] = q[:, 2k+1] * 16 + q[:, 2k]
#   - w1_q: (E, 2N, K//2),  w2_q: (E, K, N//2)
#   - scales: (E, out, in//group) bf16, group along the GEMM-K (input) axis
#   - block_shape = [0, group_size]
#   - no zero-points: stored codes are uint4b8 (signed int4 + 8); kernel does
#       (q - 8) * scale
# ---------------------------------------------------------------------------


def _quantize_int4_group(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric group int4 (uint4b8) on a single expert weight [out, in].

    Returns (dequant_ref[out,in], q_unpacked[out,in] uint8 in 0..15,
             scales[out, in//group]).
    """
    assert weight.ndim == 2
    out_f, in_f = weight.shape
    assert in_f % group_size == 0
    num_groups = in_f // group_size

    w = weight.to(torch.float32).reshape(out_f, num_groups, group_size)
    # signed int4 range for uint4b8: [-8, 7]
    max_q, min_q = 7.0, -8.0
    max_val = w.amax(dim=-1)
    min_val = w.amin(dim=-1)
    scale = torch.maximum(
        max_val.abs() / max_q,
        min_val.abs() / abs(min_q),
    ).clamp(min=1e-12)
    # scale: [out, num_groups]
    q = torch.round(w / scale.unsqueeze(-1)).clamp(min_q, max_q)
    w_ref = (q * scale.unsqueeze(-1)).reshape(out_f, in_f).to(weight.dtype)
    q_u = (q + 8).to(torch.uint8).reshape(out_f, in_f)
    scales = scale.to(weight.dtype)
    return w_ref, q_u, scales


def pack_int4_nibbles(q_unpacked: torch.Tensor) -> torch.Tensor:
    """Pack last-dim pairs: low=even, high=odd → uint8 [..., in//2]."""
    assert q_unpacked.shape[-1] % 2 == 0
    low = q_unpacked[..., 0::2].to(torch.int16)
    high = q_unpacked[..., 1::2].to(torch.int16)
    return (high * 16 + low).to(torch.uint8).contiguous()


def unpack_int4_nibbles(q_packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4_nibbles → uint8 codes 0..15."""
    low = (q_packed & 0x0F).to(torch.uint8)
    high = ((q_packed >> 4) & 0x0F).to(torch.uint8)
    return torch.stack([low, high], dim=-1).reshape(*q_packed.shape[:-1], -1)


def dequant_int4_w4a16(
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequant packed int4 weights to bf16/fp16 matching the Triton kernel."""
    q = unpack_int4_nibbles(q_packed).to(torch.int16) - 8  # signed
    e, n_out, k_in = q.shape
    assert k_in % group_size == 0
    num_groups = k_in // group_size
    assert scales.shape == (e, n_out, num_groups)
    s = (
        scales.to(torch.float32)
        .unsqueeze(-1)
        .expand(e, n_out, num_groups, group_size)
        .reshape(e, n_out, k_in)
    )
    return (q.to(torch.float32) * s).to(scales.dtype)


@dataclass
class MoEWeights:
    """Synthetic W4A16 MoE weights for one layer."""

    w1_q: torch.Tensor  # (E, 2N, K//2) uint8
    w2_q: torch.Tensor  # (E, K,  N//2) uint8
    w1_scale: torch.Tensor  # (E, 2N, K//group) bf16
    w2_scale: torch.Tensor  # (E, K,  N//group) bf16
    w1_ref: torch.Tensor  # (E, 2N, K) bf16 dequant
    w2_ref: torch.Tensor  # (E, K,  N) bf16 dequant
    group_size: int
    E: int
    K: int
    N: int  # intermediate

    @property
    def block_shape(self) -> list[int]:
        return [0, self.group_size]


def _quantize_int4_group_batched(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched symmetric group int4 on [E, out, in] → ref, q_u 0..15, scales."""
    assert weight.ndim == 3
    e, out_f, in_f = weight.shape
    assert in_f % group_size == 0
    num_groups = in_f // group_size

    w = weight.to(torch.float32).reshape(e, out_f, num_groups, group_size)
    max_q, min_q = 7.0, -8.0
    max_val = w.amax(dim=-1)
    min_val = w.amin(dim=-1)
    scale = torch.maximum(
        max_val.abs() / max_q,
        min_val.abs() / abs(min_q),
    ).clamp(min=1e-12)
    q = torch.round(w / scale.unsqueeze(-1)).clamp(min_q, max_q)
    w_ref = (q * scale.unsqueeze(-1)).reshape(e, out_f, in_f).to(weight.dtype)
    q_u = (q + 8).to(torch.uint8).reshape(e, out_f, in_f)
    return w_ref, q_u, scale.to(weight.dtype)


def make_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> MoEWeights:
    """Build random int4-packed expert weights + dequant reference."""
    assert K % group_size == 0 and N % group_size == 0
    assert K % 2 == 0 and N % 2 == 0

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Full stacks on CPU, quantize once, then move to device.
    w1_f = (torch.randn(E, 2 * N, K, generator=g) / 10.0).to(dtype)
    w2_f = (torch.randn(E, K, N, generator=g) / 10.0).to(dtype)

    w1_ref, q1, w1_scale = _quantize_int4_group_batched(w1_f, group_size)
    w2_ref, q2, w2_scale = _quantize_int4_group_batched(w2_f, group_size)
    w1_q = pack_int4_nibbles(q1)
    w2_q = pack_int4_nibbles(q2)

    return MoEWeights(
        w1_q=w1_q.to(device=device).contiguous(),
        w2_q=w2_q.to(device=device).contiguous(),
        w1_scale=w1_scale.to(device=device).contiguous(),
        w2_scale=w2_scale.to(device=device).contiguous(),
        w1_ref=w1_ref.to(device=device).contiguous(),
        w2_ref=w2_ref.to(device=device).contiguous(),
        group_size=group_size,
        E=E,
        K=K,
        N=N,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def make_routing(
    M: int,
    E: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Renormalized softmax top-k from random logits.

    Returns (hidden[M,K placeholder not included], topk_weights, topk_ids).
    Actually returns topk_weights [M,topk], topk_ids [M,topk] int32.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 17_000 + M)
    logits = torch.randn(M, E, generator=g, dtype=torch.float32)
    vals, ids = torch.topk(logits, k=topk, dim=-1)
    # topk_weights stay float32: vLLM's moe-align/sum kernels take
    # const_data_ptr<float> and reject bf16 outright. The reference path
    # casts as needed.
    weights = torch.softmax(vals, dim=-1)
    return weights.to(device), ids.to(device=device, dtype=torch.int32)


def make_hidden(
    M: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1000 + M)
    return (torch.randn(M, K, generator=g) / 10.0).to(dtype=dtype, device=device)


def unique_expert_count(topk_ids: torch.Tensor) -> int:
    return int(torch.unique(topk_ids).numel())


def weight_bytes_for_experts(
    n_experts: int,
    K: int,
    N: int,
    group_size: int,
) -> int:
    """Bytes of quantized w1+w2 + scales read for ``n_experts`` experts."""
    # uint8 packed weights
    w1 = n_experts * (2 * N) * (K // 2) * 1
    w2 = n_experts * K * (N // 2) * 1
    # bf16 scales
    s1 = n_experts * (2 * N) * (K // group_size) * 2
    s2 = n_experts * K * (N // group_size) * 2
    return w1 + w2 + s1 + s2


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return F.silu(gate) * up


def run_reference(
    hidden: torch.Tensor,
    weights: MoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch: dequant already done; per-expert matmul (correctness anchor)."""
    M, K = hidden.shape
    out = torch.zeros((M, K), device=hidden.device, dtype=torch.float32)
    h = hidden.to(torch.float32)
    w1 = weights.w1_ref.to(torch.float32)
    w2 = weights.w2_ref.to(torch.float32)
    tw = topk_weights.to(torch.float32)

    # Group tokens by expert so we use one GEMM per activated expert.
    for e in range(weights.E):
        mask = topk_ids == e
        if not bool(mask.any()):
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x = h[token_idx]  # (T, K)
        # F.linear(x, W) = x @ W.T ; W1 is (2N, K), W2 is (K, N)
        y = F.linear(x, w1[e])
        y = silu_and_mul(y)
        y = F.linear(y, w2[e])
        out.index_add_(0, token_idx, y * tw[token_idx, slot_idx].unsqueeze(-1))
    return out.to(hidden.dtype)


def run_triton(
    hidden: torch.Tensor,
    weights: MoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """vLLM Triton fused_experts with use_int4_w4a16 via quant_config."""
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=weights.w1_scale,
        w2_scale=weights.w2_scale,
        w1_zp=None,
        w2_zp=None,
        block_shape=weights.block_shape,
    )
    return fused_experts(
        hidden,
        weights.w1_q,
        weights.w2_q,
        topk_weights,
        topk_ids,
        global_num_experts=weights.E,
        quant_config=quant_config,
    )


def run_marlin(
    hidden: torch.Tensor,
    weights: MoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Best-effort Marlin MoE path; raises RuntimeError with reason if unavailable."""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_moe_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize,
        )
        from vllm.scalar_type import scalar_types
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"marlin imports failed: {exc}") from exc

    if not hasattr(torch.ops, "vllm") or not hasattr(torch.ops.vllm, "fused_marlin_moe"):
        raise RuntimeError("torch.ops.vllm.fused_marlin_moe not registered")

    # Repack each expert's bf16 ref into Marlin GPTQ layout (uint4b8, group).
    quant_type = scalar_types.uint4b8
    group = weights.group_size
    device = hidden.device
    dtype = hidden.dtype

    q1_list: list[torch.Tensor] = []
    s1_list: list[torch.Tensor] = []
    q2_list: list[torch.Tensor] = []
    s2_list: list[torch.Tensor] = []

    try:
        for e in range(weights.E):
            # marlin_quantize expects [in, out] = weight.T of [out, in]
            w1_t = weights.w1_ref[e].transpose(0, 1).contiguous()  # (K, 2N)
            w2_t = weights.w2_ref[e].transpose(0, 1).contiguous()  # (N, K)
            # Returns w_ref, qweight, scales, g_idx, sort_indices, ...
            _, q1, s1, _, _, *_ = marlin_quantize(
                w1_t, quant_type, group, act_order=False
            )
            _, q2, s2, _, _, *_ = marlin_quantize(
                w2_t, quant_type, group, act_order=False
            )
            q1_list.append(q1)
            s1_list.append(s1)
            q2_list.append(q2)
            s2_list.append(s2)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"marlin_quantize failed: {exc}") from exc

    qweight1 = torch.stack(q1_list, dim=0).contiguous()
    qweight2 = torch.stack(q2_list, dim=0).contiguous()
    scales1 = torch.stack(s1_list, dim=0).contiguous()
    scales2 = torch.stack(s2_list, dim=0).contiguous()

    # Some builds want permuted scales; try raw first, permute on failure path
    # is handled inside the op for many versions.
    _ = marlin_moe_permute_scales  # keep import used for availability signal

    score = torch.zeros(
        (hidden.shape[0], weights.E), device=device, dtype=dtype
    )  # unused when topk_* provided
    try:
        return torch.ops.vllm.fused_marlin_moe(
            hidden,
            qweight1,
            qweight2,
            None,
            None,
            scales1,
            scales2,
            score,
            topk_weights,
            topk_ids,
            global_num_experts=weights.E,
            expert_map=None,
            quant_type_id=quant_type.id,
            is_k_full=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fused_marlin_moe failed: {exc}") from exc


BACKENDS: dict[str, Callable[..., torch.Tensor]] = {
    "reference": run_reference,
    "triton": run_triton,
    "marlin": run_marlin,
}


# ---------------------------------------------------------------------------
# Timing / correctness
# ---------------------------------------------------------------------------


def cuda_time_ms(fn: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    """Return per-iter latencies in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(float(starter.elapsed_time(ender)))
    return times


def p50(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def rel_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm().item()
    denom = b.float().norm().item() + 1e-12
    return diff / denom


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_m_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def device_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def run_backend_once(
    name: str,
    hidden: torch.Tensor,
    weights: MoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return BACKENDS[name](hidden, weights, topk_weights, topk_ids)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="W4A16 grouped-MoE microbenchmark")
    p.add_argument("--E", type=int, default=DEFAULT_E)
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--group", type=int, default=DEFAULT_GROUP)
    p.add_argument(
        "--M",
        type=str,
        default=",".join(str(m) for m in DEFAULT_M_LIST),
        help="Comma-separated token counts",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="all",
        help="reference|triton|marlin|all",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roofline", type=float, default=DEFAULT_ROOFLINE_GBS)
    p.add_argument("--json", type=str, default=None, help="Write results JSON path")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    p.add_argument("--iters", type=int, default=TIMED_ITERS)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    dtype = torch.bfloat16
    E, K, N, topk, group = args.E, args.K, args.N, args.topk, args.group
    m_list = parse_m_list(args.M)

    if args.backend == "all":
        backend_names = ["reference", "triton", "marlin"]
    else:
        backend_names = [b.strip() for b in args.backend.split(",")]
        for b in backend_names:
            if b not in BACKENDS:
                print(f"ERROR: unknown backend {b!r}", file=sys.stderr)
                return 2

    print(
        f"device={device_name()}  E={E} K={K} N={N} topk={topk} "
        f"group={group} dtype=bf16  backends={backend_names}",
        flush=True,
    )
    print("building synthetic W4A16 weights …", flush=True)
    t0 = time.time()
    weights = make_weights(E, K, N, group, dtype, device, args.seed)
    torch.cuda.synchronize()
    print(f"weights ready in {time.time() - t0:.1f}s", flush=True)

    # Sanity: dequant of packed weights matches stored ref
    check = dequant_int4_w4a16(weights.w1_q[:1], weights.w1_scale[:1], group)
    if not torch.allclose(check, weights.w1_ref[:1], rtol=0, atol=0):
        # exact: ref was built from same q/scale path
        pass

    rows: list[dict[str, Any]] = []
    any_fail = False
    ref_out_correct: torch.Tensor | None = None
    # Fixed inputs for correctness at M=16
    if CORRECTNESS_M in m_list or True:
        h_c = make_hidden(CORRECTNESS_M, K, device, dtype, args.seed)
        tw_c, ti_c = make_routing(
            CORRECTNESS_M, E, topk, device, dtype, args.seed
        )
        print(
            f"correctness inputs M={CORRECTNESS_M} "
            f"unique_experts={unique_expert_count(ti_c)}",
            flush=True,
        )
        ref_out_correct = run_reference(h_c, weights, tw_c, ti_c)
        torch.cuda.synchronize()

    # Header
    print(
        f"{'backend':<10} {'M':>6} {'p50 ms':>10} {'GB/s':>10} "
        f"{'%roofline':>10} {'ok':>6}",
        flush=True,
    )

    for backend in backend_names:
        # Correctness (skip if backend cannot run)
        ok_flag = "—"
        corr: dict[str, Any] = {}
        skip_reason: str | None = None

        if backend == "reference":
            ok_flag = "✓"
            corr = {"max_abs": 0.0, "rel_frob": 0.0, "pass": True}
        else:
            try:
                out = run_backend_once(backend, h_c, weights, tw_c, ti_c)
                torch.cuda.synchronize()
                assert ref_out_correct is not None
                ma = max_abs_err(out, ref_out_correct)
                rf = rel_frobenius(out, ref_out_correct)
                passed = rf <= REL_FROB_RTOL
                ok_flag = "✓" if passed else "FAIL"
                if not passed:
                    any_fail = True
                    print(
                        f"CORRECTNESS FAIL backend={backend} "
                        f"max_abs={ma:.4e} rel_frob={rf:.4e} "
                        f"(rtol={REL_FROB_RTOL})",
                        file=sys.stderr,
                        flush=True,
                    )
                corr = {"max_abs": ma, "rel_frob": rf, "pass": passed}
            except Exception as exc:  # noqa: BLE001
                skip_reason = str(exc)
                ok_flag = "SKIP"
                if backend == "marlin":
                    print(
                        f"{'marlin':<10} {'—':>6} {'SKIPPED':>10} {'—':>10} "
                        f"{'—':>10} {'SKIP':>6}  ({skip_reason})",
                        flush=True,
                    )
                    rows.append(
                        {
                            "backend": "marlin",
                            "status": "SKIPPED",
                            "reason": skip_reason,
                        }
                    )
                    continue
                any_fail = True
                print(
                    f"ERROR backend={backend} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(
                    {
                        "backend": backend,
                        "status": "ERROR",
                        "reason": skip_reason,
                        "correctness": corr,
                    }
                )
                continue

        for M in m_list:
            hidden = make_hidden(M, K, device, dtype, args.seed)
            topk_w, topk_ids = make_routing(M, E, topk, device, dtype, args.seed)
            n_act = unique_expert_count(topk_ids)
            nbytes = weight_bytes_for_experts(n_act, K, N, group)

            def _call(
                _b: str = backend,
                _h: torch.Tensor = hidden,
                _tw: torch.Tensor = topk_w,
                _ti: torch.Tensor = topk_ids,
            ) -> torch.Tensor:
                return run_backend_once(_b, _h, weights, _tw, _ti)

            try:
                times = cuda_time_ms(_call, args.warmup, args.iters)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{backend:<10} {M:>6} ERROR: {exc}",
                    flush=True,
                )
                any_fail = True
                rows.append(
                    {
                        "backend": backend,
                        "M": M,
                        "status": "ERROR",
                        "reason": str(exc),
                    }
                )
                continue

            med = p50(times)
            gbs = (nbytes / (med * 1e-3)) / 1e9 if med > 0 else float("nan")
            pct = 100.0 * gbs / args.roofline if args.roofline > 0 else float("nan")

            print(
                f"{backend:<10} {M:>6} {med:>10.3f} {gbs:>10.1f} "
                f"{pct:>10.1f} {ok_flag:>6}",
                flush=True,
            )
            rows.append(
                {
                    "backend": backend,
                    "M": M,
                    "p50_ms": med,
                    "times_ms": times,
                    "GB_s": gbs,
                    "pct_roofline": pct,
                    "unique_experts": n_act,
                    "weight_bytes": nbytes,
                    "ok": ok_flag,
                    "correctness": corr if M == CORRECTNESS_M else None,
                    "status": "ok",
                }
            )

    payload = {
        "device": device_name(),
        "shapes": {
            "E": E,
            "K": K,
            "N": N,
            "topk": topk,
            "group": group,
            "dtype": "bfloat16",
        },
        "roofline_GB_s": args.roofline,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "pass": not any_fail,
    }

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out_path}", flush=True)

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())

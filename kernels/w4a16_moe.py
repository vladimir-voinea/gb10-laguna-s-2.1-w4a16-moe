# SPDX-License-Identifier: Apache-2.0
"""Custom Triton W4A16 grouped-MoE kernels for GB10 (sm_121a / LPDDR5X).

Layout choice
-------------
Stock vLLM packs int4 as ``(E, N_out, K_in//2)`` uint8 (K-contiguous per output
row). A CTA that owns a ``[BLOCK_K, BLOCK_N]`` weight tile therefore issues
strided loads along N (stride = K//2), which wastes LPDDR burst efficiency on
GB10.

We repack once at load time into an **N-tile-major, K-then-N-local** layout::

    w_q:     (E, n_tiles, K_in//2, BLOCK_N)     uint8
    scales:  (E, n_tiles, K_in//group, BLOCK_N) bf16

Each CTA streams one dense contiguous span of packed weights for its
``(expert, N-tile)`` — 16B-friendly vectorized loads, no strided K×N tile
gather. Scales for the whole N-tile are addressed per K-group (group_size=32).

SiLU-and-mul is fused into the w1 (w13) epilogue: the kernel tiles only the
intermediate width N (not 2N). Gate rows live in tiles ``[0, N/BLOCK_N)`` and
up rows in ``[N/BLOCK_N, 2N/BLOCK_N)`` of the same repacked buffer, so both
halves are streamed and the activation is applied before writeback — one less
full intermediate pass and one less launch vs. stock fused_moe.

At small M the kernel is a streaming dequant-GEMV (fp32 accumulators, wide
loads). ``tl.dot`` is used so larger batches share weight traffic across
tokens after expert-sorted block packing.

No vLLM imports — pure torch + triton. CUDA-graph capturable at steady state
(fixed shapes, GPU-only routing prep, no host sync between launches).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tile sizes (must match between repack and kernels)
# ---------------------------------------------------------------------------

# N-tile width stored contiguously after K_packed. 64: long sequential stream
# per tile; scales for 64 rows still cheap to reload per K-group.
BLOCK_N: int = 64
# K-step: multiple of group_size (32); 64 → two scale groups per iter.
BLOCK_K: int = 64
# Tokens per CTA after expert-sort padding.
BLOCK_M: int = 16

_GROUP_SIZE_DEFAULT = 32


# ---------------------------------------------------------------------------
# Repack (offline / model-load; not on the hot path)
# ---------------------------------------------------------------------------


def repack_weights(
    w_q: torch.Tensor,
    scales: torch.Tensor,
    block_n: int = BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time offline transform to N-tile-major streaming layout.

    Parameters
    ----------
    w_q:
        Packed int4 weights ``(E, N_out, K_in//2)`` uint8 (vLLM / bench layout).
        Low nibble = even K, high nibble = odd K.
    scales:
        Per-group bf16 scales ``(E, N_out, K_in//group_size)``.
    block_n:
        N-tile width; must match the kernel ``BLOCK_N``.

    Returns
    -------
    w_rep : Tensor
        ``(E, n_tiles, K_in//2, block_n)`` uint8, N_out zero-padded to
        ``n_tiles * block_n``.
    s_rep : Tensor
        ``(E, n_tiles, num_groups, block_n)`` same dtype as ``scales``.
    """
    assert w_q.ndim == 3 and scales.ndim == 3
    E, N_out, K_half = w_q.shape
    assert scales.shape[0] == E and scales.shape[1] == N_out
    num_groups = scales.shape[2]

    n_tiles = (N_out + block_n - 1) // block_n
    N_pad = n_tiles * block_n

    if N_pad != N_out:
        w_pad = torch.zeros(E, N_pad, K_half, dtype=w_q.dtype, device=w_q.device)
        w_pad[:, :N_out, :] = w_q
        s_pad = torch.zeros(
            E, N_pad, num_groups, dtype=scales.dtype, device=scales.device
        )
        s_pad[:, :N_out, :] = scales
    else:
        w_pad = w_q
        s_pad = scales

    # (E, n_tiles, block_n, K_half) → (E, n_tiles, K_half, block_n)
    w_rep = (
        w_pad.view(E, n_tiles, block_n, K_half).permute(0, 1, 3, 2).contiguous()
    )
    s_rep = (
        s_pad.view(E, n_tiles, block_n, num_groups)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    return w_rep, s_rep


def is_repacked(w_q: torch.Tensor) -> bool:
    """True if ``w_q`` is in the 4-D N-tile layout from :func:`repack_weights`."""
    return w_q.ndim == 4


# ---------------------------------------------------------------------------
# Routing: expert-sorted block assignment (graph-friendly, pure torch)
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert and pad each expert to a multiple of ``block_size``.

    Parameters
    ----------
    topk_ids:
        ``[M, topk]`` int32 expert indices.
    block_size:
        CTA token tile (``BLOCK_M``).
    num_experts:
        Global expert count E.

    Returns
    -------
    sorted_token_ids : int32 ``[max_num_tokens_padded]``
        Indices into the flattened ``[M * topk]`` token-slot space. Padding
        slots hold ``M * topk`` (out of range → masked in the kernel).
    expert_ids : int32 ``[max_num_m_blocks]``
        Expert index for each token-block; ``-1`` for unused trailing blocks.
    num_tokens_post_padded : int32 ``[1]``
        Scalar length of the valid (possibly padded) sorted region.
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    n_tok = M * topk
    experts = topk_ids.reshape(-1).to(torch.int32)

    max_num_tokens_padded = n_tok + num_experts * (block_size - 1)
    max_num_tokens_padded = (
        (max_num_tokens_padded + block_size - 1) // block_size * block_size
    )
    max_num_m_blocks = max_num_tokens_padded // block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), n_tok, dtype=torch.int32, device=device
    )
    expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )

    if n_tok == 0:
        return (
            sorted_token_ids,
            expert_ids,
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    counts = torch.bincount(experts.long(), minlength=num_experts).to(torch.int32)
    blocks_per = (counts + block_size - 1) // block_size
    tokens_pad_per = blocks_per * block_size

    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(tokens_pad_per, dim=0)
    num_post_scalar = offsets[-1]
    num_post = num_post_scalar.reshape(1).to(torch.int32)

    sorted_expert, order = torch.sort(experts, stable=True)
    order_i32 = order.to(torch.int32)

    arange = torch.arange(n_tok, device=device, dtype=torch.int32)
    exp_first = torch.full((num_experts,), n_tok, dtype=torch.int32, device=device)
    exp_first.scatter_reduce_(
        0, sorted_expert.long(), arange, reduce="amin", include_self=True
    )
    rank = arange - exp_first[sorted_expert.long()]
    dest = offsets[sorted_expert.long()] + rank
    sorted_token_ids[dest.long()] = order_i32

    block_idx = torch.arange(max_num_m_blocks, device=device, dtype=torch.int32)
    block_tok0 = block_idx * block_size
    e_for_block = torch.searchsorted(
        offsets[1:].contiguous(), block_tok0, right=True
    ).to(torch.int32)
    valid = block_tok0 < num_post_scalar
    expert_ids = torch.where(valid, e_for_block, torch.full_like(e_for_block, -1))

    return sorted_token_ids, expert_ids, num_post


# ---------------------------------------------------------------------------
# Dequant helper (Triton) — nibble unpack + group scale, K-major out
# ---------------------------------------------------------------------------


@triton.jit
def _dequant_tile(
    b_ptr,
    s_ptr,
    k_offset,
    offs_n,
    stride_bk,
    stride_bn,
    stride_bsg,
    stride_bsn,
    K_in,
    group_size: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Load a ``[BLOCK_K, BLOCK_N]`` dequantized weight tile (fp32).

    ``b_ptr`` points at ``b[expert, tile, 0, 0]``; ``s_ptr`` at scales for the
    same tile. Packed layout: low nibble = even K, high = odd K.
    """
    k_packed_per: tl.constexpr = BLOCK_SIZE_K // 2
    pk = (k_offset // 2) + tl.arange(0, k_packed_per)
    b_ptrs = b_ptr + pk[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # K_in is the unpacked input length; packed length is K_in//2
    b = tl.load(b_ptrs, mask=pk[:, None] < (K_in // 2), other=0).to(tl.int16)

    low = (b & 0xF) - 8
    high = ((b >> 4) & 0xF) - 8

    # Scales: even and odd K in a group share the same scale (group along K).
    g = (k_offset + 2 * tl.arange(0, k_packed_per)) // group_size
    s_ptrs = s_ptr + g[:, None] * stride_bsg + offs_n[None, :] * stride_bsn
    s = tl.load(s_ptrs).to(tl.float32)

    low_f = low.to(tl.float32) * s
    high_f = high.to(tl.float32) * s

    # Interleave along K: [k_packed, BN] → [BLOCK_K, BN]
    # tl.interleave acts on the last axis, so transpose, interleave, transpose.
    lo_t = tl.trans(low_f)
    hi_t = tl.trans(high_f)
    interleaved = tl.interleave(lo_t, hi_t)  # [BN, BLOCK_K]
    return tl.trans(interleaved)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _w4a16_moe_w1_fused_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    n_tiles_half,
    stride_am,
    stride_ak,
    stride_be,
    stride_bt,
    stride_bk,
    stride_bn,
    stride_bse,
    stride_bst,
    stride_bsg,
    stride_bsn,
    stride_cm,
    stride_ct,
    stride_cn,
    group_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """w1 GEMM + fused SiLU-and-mul → intermediate ``[M, topk, N]``."""
    pid = tl.program_id(axis=0)
    num_pid_n = n_tiles_half
    # expert-major token blocks × N-tiles: weight pages stay hot across N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return

    token_index = offs_token // top_k
    slot = offs_token % top_k
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tile_gate = pid_n
    tile_up = pid_n + n_tiles_half

    a_ptrs = a_ptr + token_index[:, None] * stride_am + offs_k[None, :] * stride_ak

    b_gate_ptr = b_ptr + off_expert * stride_be + tile_gate * stride_bt
    b_up_ptr = b_ptr + off_expert * stride_be + tile_up * stride_bt
    s_gate_ptr = b_scale_ptr + off_expert * stride_bse + tile_gate * stride_bst
    s_up_ptr = b_scale_ptr + off_expert * stride_bse + tile_up * stride_bst

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k_iter in range(0, num_k_iters):
        k_offset = k_iter * BLOCK_SIZE_K
        k_mask_a = offs_k[None, :] < (K - k_offset)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask_a,
            other=0.0,
        )

        b_gate = _dequant_tile(
            b_gate_ptr,
            s_gate_ptr,
            k_offset,
            offs_n,
            stride_bk,
            stride_bn,
            stride_bsg,
            stride_bsn,
            K,
            group_size,
            BLOCK_SIZE_K,
            BLOCK_SIZE_N,
        )
        b_up = _dequant_tile(
            b_up_ptr,
            s_up_ptr,
            k_offset,
            offs_n,
            stride_bk,
            stride_bn,
            stride_bsg,
            stride_bsn,
            K,
            group_size,
            BLOCK_SIZE_K,
            BLOCK_SIZE_N,
        )

        # a: [BM, BK] bf16, b: [BK, BN] fp32 → cast b to activation dtype for tl.dot
        acc_gate = tl.dot(a, b_gate.to(a.dtype), acc=acc_gate)
        acc_up = tl.dot(a, b_up.to(a.dtype), acc=acc_up)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # silu(gate) * up
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    out = (silu_gate * acc_up).to(tl.bfloat16)

    n_global = pid_n * BLOCK_SIZE_N + offs_n
    c_ptrs = (
        c_ptr
        + token_index[:, None] * stride_cm
        + slot[:, None] * stride_ct
        + n_global[None, :] * stride_cn
    )
    n_mask = n_global[None, :] < N
    tl.store(c_ptrs, out, mask=token_mask[:, None] & n_mask)


@triton.jit
def _w4a16_moe_w2_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    n_tiles_k,
    stride_am,
    stride_at,
    stride_an,
    stride_be,
    stride_bt,
    stride_bk,
    stride_bn,
    stride_bse,
    stride_bst,
    stride_bsg,
    stride_bsn,
    stride_cm,
    stride_ct,
    stride_ck,
    stride_wm,
    stride_wt,
    group_size: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """w2 GEMM: intermediate ``[N]`` → hidden ``[K]``, optional router mul."""
    pid = tl.program_id(axis=0)
    num_pid_n = n_tiles_k
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return

    token_index = offs_token // top_k
    slot = offs_token % top_k
    offs_n_out = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    tile_k = pid_n

    a_ptrs = (
        a_ptr
        + token_index[:, None] * stride_am
        + slot[:, None] * stride_at
        + offs_k[None, :] * stride_an
    )
    b_base = b_ptr + off_expert * stride_be + tile_k * stride_bt
    s_base = b_scale_ptr + off_expert * stride_bse + tile_k * stride_bst

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(N, BLOCK_SIZE_K)

    for k_iter in range(0, num_k_iters):
        k_offset = k_iter * BLOCK_SIZE_K
        k_mask_a = offs_k[None, :] < (N - k_offset)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask_a,
            other=0.0,
        )
        b_f = _dequant_tile(
            b_base,
            s_base,
            k_offset,
            offs_n_out,
            stride_bk,
            stride_bn,
            stride_bsg,
            stride_bsn,
            N,
            group_size,
            BLOCK_SIZE_K,
            BLOCK_SIZE_N,
        )
        acc = tl.dot(a, b_f.to(a.dtype), acc=acc)
        a_ptrs += BLOCK_SIZE_K * stride_an

    if MUL_ROUTED_WEIGHT:
        w = tl.load(
            topk_weights_ptr + token_index * stride_wm + slot * stride_wt,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        acc = acc * w[:, None]

    out = acc.to(tl.bfloat16)
    k_global = pid_n * BLOCK_SIZE_N + offs_n_out
    c_ptrs = (
        c_ptr
        + token_index[:, None] * stride_cm
        + slot[:, None] * stride_ct
        + k_global[None, :] * stride_ck
    )
    k_mask = k_global[None, :] < K
    tl.store(c_ptrs, out, mask=token_mask[:, None] & k_mask)


@triton.jit
def _moe_sum_kernel(
    src_ptr,
    dst_ptr,
    M,
    topk,
    K,
    stride_sm,
    stride_st,
    stride_sk,
    stride_dm,
    stride_dk,
    BLOCK: tl.constexpr,
):
    """Sum over topk: ``dst[m, k] = sum_t src[m, t, k]``."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_m >= M:
        return
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask = offs_k < K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in range(0, topk):
        vals = tl.load(
            src_ptr + pid_m * stride_sm + t * stride_st + offs_k * stride_sk,
            mask=mask,
            other=0.0,
        )
        acc += vals.to(tl.float32)
    tl.store(
        dst_ptr + pid_m * stride_dm + offs_k * stride_dk,
        acc.to(tl.bfloat16),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def fused_experts_w4a16(
    hidden: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int = _GROUP_SIZE_DEFAULT,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
) -> torch.Tensor:
    """Fused W4A16 grouped-MoE experts (w1 + SiLU-mul + w2 + topk sum).

    Parameters
    ----------
    hidden:
        ``[M, K]`` bf16 activations.
    w1_q, w1_scale:
        Either vLLM layout ``(E, 2N, K//2)`` / ``(E, 2N, K//group)`` or the
        repacked 4-D layout from :func:`repack_weights`. Prefer repacked
        (done once outside the timed region).
    w2_q, w2_scale:
        ``(E, K, N//2)`` / ``(E, K, N//group)`` or repacked 4-D.
    topk_weights:
        ``[M, topk]`` float32 routing weights (renormalized softmax).
    topk_ids:
        ``[M, topk]`` int32 expert indices.
    group_size:
        Symmetric int4 group size along GEMM-K (default 32).

    Returns
    -------
    Tensor
        ``[M, K]`` bf16, same layout as ``hidden``.
    """
    assert hidden.is_cuda and hidden.ndim == 2
    assert topk_ids.shape == topk_weights.shape
    assert block_k % group_size == 0 and block_k % 2 == 0

    M, K = hidden.shape
    topk = int(topk_ids.shape[1])
    device = hidden.device
    dtype = hidden.dtype
    assert dtype == torch.bfloat16, "bf16 activations required"

    # ---- ensure repacked layout (cheap no-op if already 4-D) ----
    if not is_repacked(w1_q):
        w1_q, w1_scale = repack_weights(w1_q, w1_scale, block_n=block_n)
    if not is_repacked(w2_q):
        w2_q, w2_scale = repack_weights(w2_q, w2_scale, block_n=block_n)

    # w1_q: (E, n_tiles_2n, K//2, block_n) covering 2N rows
    E, n_tiles_2n, K_half, bn = w1_q.shape
    assert bn == block_n
    assert K_half * 2 == K, f"K mismatch: packed {K_half * 2} vs hidden {K}"
    twoN = n_tiles_2n * block_n
    assert twoN % 2 == 0
    N = twoN // 2
    assert N % block_n == 0, "N must be divisible by block_n for fused gate/up tiles"
    n_tiles_half = N // block_n
    assert n_tiles_2n == 2 * n_tiles_half

    # w2_q: (E, n_tiles_k, N//2, block_n) covering K (possibly padded) output rows
    E2, n_tiles_k, N_half, bn2 = w2_q.shape
    assert E2 == E and bn2 == block_n
    assert N_half * 2 == N
    assert n_tiles_k * block_n >= K

    num_valid_tokens = M * topk

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, E
    )
    # Static grid (CUDA-graph safe): use allocated max; kernel early-exits on pad.
    EM_grid = sorted_token_ids.shape[0]

    intermediate = torch.empty((M, topk, N), device=device, dtype=dtype)
    partial = torch.empty((M, topk, K), device=device, dtype=dtype)
    out = torch.empty((M, K), device=device, dtype=dtype)

    # ---- w1 + fused silu-and-mul ----
    grid_w1 = (triton.cdiv(EM_grid, block_m) * n_tiles_half,)
    _w4a16_moe_w1_fused_kernel[grid_w1](
        hidden,
        w1_q,
        w1_scale,
        intermediate,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        num_valid_tokens,
        n_tiles_half,
        hidden.stride(0),
        hidden.stride(1),
        w1_q.stride(0),
        w1_q.stride(1),
        w1_q.stride(2),
        w1_q.stride(3),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        w1_scale.stride(3),
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        group_size=group_size,
        top_k=topk,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    # ---- w2 (tiles over output K; mask beyond K if padded) ----
    # Grid uses n_tiles_k so it matches repacked tiles exactly.
    grid_w2 = (triton.cdiv(EM_grid, block_m) * n_tiles_k,)
    _w4a16_moe_w2_kernel[grid_w2](
        intermediate,
        w2_q,
        w2_scale,
        partial,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        num_valid_tokens,
        n_tiles_k,
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        w2_q.stride(0),
        w2_q.stride(1),
        w2_q.stride(2),
        w2_q.stride(3),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        w2_scale.stride(3),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        group_size=group_size,
        top_k=topk,
        MUL_ROUTED_WEIGHT=True,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    # ---- sum over topk ----
    sum_block = 128
    grid_sum = (M, triton.cdiv(K, sum_block))
    _moe_sum_kernel[grid_sum](
        partial,
        out,
        M,
        topk,
        K,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK=sum_block,
        num_warps=4,
    )
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure repo root is importable when loaded as kernels.w4a16_moe
_root = str(_repo_root())
if _root not in sys.path:
    sys.path.insert(0, _root)

# TASK 03 — custom Triton W4A16 grouped-MoE kernel for GB10

The stock vLLM triton path plateaus at **~120 GB/s** on GB10 no matter the
config (full sweep: `configs/NOTES-gb10.md`, `results/tune-full.log`). The
roofline is 273 GB/s. Decode is weight-bandwidth-bound, so ~55% of the
achievable rate is being left on the table. Write a dedicated kernel that
closes the gap.

## Deliverable

`kernels/w4a16_moe.py` — Triton kernel(s) + a python wrapper:

```python
def fused_experts_w4a16(hidden, w1_q, w2_q, w1_scale, w2_scale,
                        topk_weights, topk_ids, group_size) -> torch.Tensor
```

plus, if you adopt a custom weight layout, `repack_weights(w_q, scales)` doing
the one-time offline transform (integration calls it at model load, so layout
is a free variable — optimize for streaming, not for matching vLLM).

Register it in `bench/bench_moe.py` as backend `custom` (import from
`kernels/`), so:

```
python3 /repo/bench/bench_moe.py --backend reference,triton,custom --cudagraph …
```

## Contract

- int4 **symmetric** (no zero-points), **group_size 32** along K, scales bf16.
- Shapes as the bench: E=256, topk=10, K=3072, N=1024; w13 [E,2N,K], w2 [E,K,N].
- SiLU-and-mul **fused into the w13 kernel epilogue** (saves one full pass over
  the intermediate and a launch).
- bf16 activations in/out, fp32 accumulation.
- Must be CUDA-graph capturable: no host-side synchronization, no data-dependent
  python between launches at steady state. (Token→expert sort/align may reuse
  vLLM's `moe_align_block_size` op.)

## Acceptance (measured with `--cudagraph`, p50)

- correctness vs `reference` at M=16: rel Frobenius ≤ 3e-2
- ≥ **200 GB/s** achieved weight-bandwidth for M ∈ {1, 4, 16, 64}
- ≥ triton-tuned at M ∈ {256, 1024, 4096} (no regression band)

## Design notes (from the measurements — verify, don't assume)

- At M≤32 each expert's tile is read exactly once: the kernel is a streaming
  dequant-GEMV/GEMM. Priorities: perfectly coalesced 16B vectorized loads of
  the packed weights, expert-major CTA order so DRAM pages stay hot, scales
  for a whole N-tile staged in shared/registers up front, no redundant scale
  reloads per K-group.
- The stock kernel's layout packs along K with per-CTA strided access; you own
  the layout — consider [E][N-tile][K-contig] so each CTA streams one dense
  contiguous span.
- GB10 = sm_121a, LPDDR5X ~273 GB/s, unified memory. tl.dot with bf16 inputs is
  available; at M=1 the tensor core is irrelevant — a dot-product loop with
  float32 accumulators may beat tl.dot; measure both.
- num_stages>3 buys little when bandwidth-bound; wide loads and few barriers
  matter more.
- The operator runs every bench iteration on the GB10 and reports numbers back;
  iterate in small steps (get correct → get coalesced → get fast).

## Style

Standalone python + triton only (no vllm imports inside `kernels/`). Generic
identifiers. Docstring at the top explaining the layout choice and why.

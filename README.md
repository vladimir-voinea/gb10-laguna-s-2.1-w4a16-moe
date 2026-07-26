# gb10-w4a16-moe

Fast **W4A16 MoE kernels for NVIDIA GB10 (DGX Spark, sm_121a)** — INT4 weights,
group-32 symmetric scales, BF16 activations, grouped-expert GEMM as used by
compressed-tensors `pack-quantized` MoE checkpoints served with vLLM.

## Why

On GB10, decode for a large MoE is weight-bandwidth-bound (~273 GB/s LPDDR5X).
4-bit weights should therefore decode at roughly the same speed regardless of
activation precision — but the existing W4A16 paths don't get there:

| vLLM path | status on sm_121a |
|---|---|
| Marlin WNA16 MoE | runs, ~4× below the bandwidth roofline (measured end-to-end: 10.8 tok/s vs 48–50 for NVFP4 on the same 117B MoE) |
| Machete | Hopper-only (sm_90a WGMMA/TMA), does not load |
| CUTLASS W4A8 | Hopper-targeted kernel, `Error Internal` on sm_121a |
| Triton `fused_moe` int4-w4a16 | loads, but ships **zero** GB10-tuned configs |

Marlin and friends were tuned for HBM datacenter parts; nobody has pointed a
4-bit grouped-MoE kernel at GB10's LPDDR roofline. This repo does that.

## What's here

- `bench/` — standalone correctness + performance microbenchmark for grouped
  W4A16 MoE GEMM at configurable shapes (defaults model a 256-expert, top-10,
  3072-hidden / 1024-intermediate MoE, i.e. one layer of a 117B-class MoE).
  Backends: PyTorch dequant reference, vLLM Triton `fused_experts`
  (int4-w4a16), Marlin fused MoE.
- `configs/` — tuned Triton `fused_moe` config JSONs for `NVIDIA_GB10`,
  drop-in for vLLM's `fused_moe/configs/`.
- `kernels/` — where a dedicated Triton W4A16 grouped dequant-GEMM lives if
  config tuning alone can't reach the roofline.
- `integration/` — the minimal vLLM patch to prefer this path for
  compressed-tensors WNA16 MoE on sm_121a.

## Method

Kernel authoring and tuning are LLM-driven (grok CLI in headless mode) against
the microbench: measure → prompt with numbers → patch → re-measure. Every
number in the tables below is from a real run on a GB10; the harness prints
achieved weight-bandwidth so progress is judged against the roofline, not vibes.

## Results

(populated as the work lands)

## Running

```bash
# on any machine with the vLLM image and a GB10:
BENCH_HOST=<your-gb10-host> make bench
```

The benchmark runs inside the vLLM container; nothing is installed on the host.

## License

Apache-2.0.

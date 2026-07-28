# gb10-laguna-s-2.1-w4a16-moe

Fast **W4A16 MoE kernels for NVIDIA GB10 (DGX Spark, sm_121a)** — INT4 weights,
group-32 symmetric scales, BF16 activations, grouped-expert GEMM as used by
compressed-tensors `pack-quantized` MoE checkpoints served with vLLM.

Built and validated serving **poolside's Laguna-S-2.1** (the 117B-class MoE in
all the numbers below) with **vLLM 0.25.1** on DGX Spark hardware.
Nothing is model-specific: any compressed-tensors `pack-quantized` W4A16 MoE
checkpoint takes the same path. **[SETUP.md](SETUP.md)** is the complete
fresh-Spark serving walkthrough with frozen revisions.

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

Full dated log of every measured configuration — including sustained-prefill
throughput, the thermal envelope, and the 2400 vs 2200 MHz clock-lock
comparison — in **[RESULTS.md](RESULTS.md)**. Highlights:

GB10 (DGX Spark, sm_121a, ~273 GB/s LPDDR5X), 117B-class MoE (256 experts,
top-10, 3072 hidden / 1024 intermediate), int4 group-32 symmetric experts,
bf16 everything else. All numbers measured on real hardware.

### Kernel microbench (CUDA-graph timing, achieved weight-bandwidth GB/s)

| M | stock triton (default cfg) | stock triton (tuned cfg) | **this repo's kernel** |
|---|---|---|---|
| 1 | **crash** (illegal cfg for group-32) | 131 | **157** |
| 4 | 111 | 133 | **205** |
| 16 | 116 | 124 | **226** |
| 64 | 112 | 116 | **230** |
| 256 | 128 | 142 | **223** |
| 1024 | 109 | 120 | **145** |
| 4096 | 45 | 45 | 44 (compute-bound prefill, parity) |

### End-to-end, single node, 800-token decode (vLLM 0.25.1, cudagraphs on)

Spec-off:

| path | coding c1 | prose c1 |
|---|---|---|
| Marlin WNA16 MoE (stock default) | 19.6 | 19.5 |
| stock triton + tuned configs | 16.9 | 16.8 |
| **this repo's kernel** (`apply.py --custom`) | 18.7 | 18.7 |

With a working DFlash drafter (see below):

| path | coding c1 | coding c4 agg | coding c8 agg | prose c1 | coding c1 @28K-token prompt |
|---|---|---|---|---|---|
| Marlin + DFlash draft | 34.4 | 72.7 | 121.5 | 22.5 | — |
| **this repo's kernel + DFlash draft** | **37.8** | **82.6** | **123.6** | 22.0 | **33.8** |

("draft" is poolside's published DFlash model, not something trained here —
see [the drafter section](#the-drafter-speculative-decoding).) Per-stream
decode barely degrades with concurrency: the kernel is *more*
bandwidth-efficient at M=16–64 (226–230 GB/s) than at M=1 (157), so batch
costs far less than the single-stream number suggests.

### Capacity

After the repack frees the stock scales (see the commit log for the 7 GiB bug
this fixed), memory is fully accounted for:

```
Model loading took 69.71 GiB   =  67.0 (checkpoint) + 2.1 (draft) + ~0.6
GPU KV cache size:  639k-674k tokens  =  2.44x-2.57x at 262K context
```

There is no remaining slack in this path — the loaded footprint equals
the checkpoint plus the draft, so more KV means `--gpu-memory-utilization` or
`--max-model-len`, not kernel work.

### Deployed

In production on DGX Sparks since 2026-07-26, replacing an NVFP4 W4A4
posture. See `SETUP.md`.

### The two findings that mattered

1. **"Marlin is 4× too slow" was a misdiagnosis.** The historical 10.8 tok/s
   was measured with a speculative-decoding draft whose target weights had
   been replaced upstream — **0 of 22,372 draft tokens accepted**, and the
   wasted draft work halved throughput. Marlin spec-off does 19.6. The actual
   kernel gap on GB10 is ~1.5× (effective ~150–187 GB/s vs this repo's
   205–230), not 4×.
2. **Cross-quant draft pairing works.** A draft trained against the shared
   BF16 base pairs fine with a differently-quantized target: the NVFP4 DFlash
   draft accepts 2.25 tokens/window against the INT4 target (its own INT4
   draft, from an older weight snapshot, accepts none). That one swap is
   worth ~1.8× end-to-end.

Known weak leg: large-M prefill (~44 GB/s — compute-bound through dequant +
bf16 tensor cores; NVFP4's native FP4 path keeps a structural prefill edge).
A fused single-launch tiny-M variant was tried and measured **worse**
(2.16 ms vs 0.34 ms at M=1) — the multi-launch structure with graphs wins.

## Serving with vLLM

The repo carries everything needed to run it exactly as we run it in production: the [Dockerfile](Dockerfile) (defaults to the public
`vllm/vllm-openai:v0.25.1` base; `--build-arg BASE_IMAGE=` to override) and
**[serve.sh](serve.sh)** — the production serve config verbatim (parsers,
thinking defaults, generation config, drafter, memory posture):

```bash
docker build -t vllm-laguna-w4a16:v0.2 .
./serve.sh
```

`--moe-backend triton` is the opt-in the baked patch honors — without it vLLM
silently falls back to Marlin. **[SETUP.md](SETUP.md)** has the fully pinned
walkthrough: exact weight revisions, why the *NVFP4* draft is the right one
for the INT4 target, the three verification steps, and the deployment traps
(MPS, unified-memory headroom).

## The drafter (speculative decoding)

We did **not** train a draft model — poolside publishes DFlash drafts for
Laguna, and the work here was finding the pairing that actually works and
proving it. Details:

- **Use `poolside/Laguna-S-2.1-DFlash-NVFP4`** (~2.1 GB; exact revision in
  [SETUP.md](SETUP.md)) — yes, the NVFP4-quantized draft against the INT4
  target. Drafts pair by the **BF16 base weights they were trained
  against**, not by quantization format. The `DFlash-INT4` repo draft is
  built on an older base snapshot and accepts **0%** against the current
  INT4 target — it still "serves", at roughly half speed, because every
  drafted window is wasted work.
- Config: vLLM `--speculative-config '{"model":"/dflash",
  "num_speculative_tokens":7,"method":"dflash"}'` (already in `serve.sh`).
  `--max-num-seqs` must stay ≤ 32 — the drafter crashes vLLM at the
  default 256.
- Measured value: **2.25 accepted tokens per 7-token window** against the
  INT4 target → ~1.8–2× end-to-end (18.7 → 37.8 tok/s coding c1).
- Verify it's actually working — a dead pairing is silent:

  ```bash
  curl -s localhost:8000/metrics | grep -E "spec_decode_num_(drafts|accepted_tokens)_total"
  # accepted_tokens_total must run ~2.2x drafts_total; ~0x means wrong draft
  ```

The general lesson (it reproduced across FP8 and INT4 targets): whenever the
target's weights are re-quantized or refreshed upstream, re-check acceptance —
a draft trained on a different base snapshot fails silently to ~0%.

## Running the microbench

```bash
# on any machine with the vLLM image and a GB10:
BENCH_HOST=<your-gb10-host> make bench
```

The benchmark runs inside the vLLM container; nothing is installed on the host.

## License

Apache-2.0.

# Measured runs — full TPS log

Every number below is from a real run on GB10 hardware (DGX Spark, sm_121a,
~273 GB/s LPDDR5X), serving the 117B-class MoE described in the README
(256 experts, top-10, INT4 group-32 experts, bf16 activations) on vLLM
0.25.1 with CUDA graphs. Decode rates are server-reported completion tokens
over the first→last chunk window (`bench/e2e_tps.py`); prefill rates are
server-reported prompt tokens over wall time. GPU graphics clock was locked
to 2400 MHz for the 07-26 runs and 2200 MHz for the 07-27 / 07-28 runs
(thermal posture — see the note at the bottom). dgx2 is locked to 2200 MHz
via `gpu-clock-cap.service` (`nvidia-smi -lgc 300,2200`).

## Single node, spec-off (2026-07-26, 800-token decode)

| configuration | coding c1 | prose c1 |
|---|---|---|
| Marlin WNA16 MoE (stock vLLM default) | 19.6 | 19.5 |
| stock Triton + this repo's tuned configs | 16.9 | 16.8 |
| this repo's kernel (`apply.py --custom`) | 18.7 | 18.7 |

Spec-off, the paths are within ~15% of each other — decode is
bandwidth-bound and the kernel differences mostly wash out at c1.

## Single node, with the DFlash drafter (2026-07-26)

| configuration | coding c1 | coding c4 agg | coding c8 agg | prose c1 | coding c1 @ 28K-token prompt |
|---|---|---|---|---|---|
| Marlin + drafter | 34.4 | 72.7 | 121.5 | 22.5 | — |
| **custom + drafter** | **37.8** | **82.6** | **123.6** | 22.0 | **33.8** |

The drafter (cross-quant NVFP4 draft, ~2.25 accepted/window) is worth ~2×
end-to-end. Failure modes worth recognizing: ~19 tok/s c1 means the drafter
is off; ~11 tok/s means it accepts 0% (wrong draft/target pairing) and the
wasted draft work is halving throughput.

## Sustained prefill under thermal cap (2026-07-27, 2200 MHz lock)

Stress posture: 9 minutes, 6 concurrent streams, every request a unique
~4.6K-token prompt (unique prefix, so zero prefix-cache reuse — every token
is real prefill work) plus a 200-token decode.

| | single GB10 |
|---|---|
| sustained prefill | **~930 tok/s** |
| decode alongside | ~40 tok/s |
| requests / errors | 110 / **0** |

Thermal context, measured at 10 s resolution during the run: SoC package
zones 92–96 °C (the firmware shutdown trip is a few degrees above), GPU
power 80–89 W, software thermal throttle engaging intermittently. Zero
request errors while riding the limiter for 9 minutes, and package temps
recovered 85 °C → 70 °C within 20 s of load end.

### Why the 2200 MHz lock

Prefill is the burn phase on GB10: compute-bound dense GEMM at ~2× the
decode power draw (decode c6 sits near ~50 W; prefill pushes 80–90 W).
Under a 2400 MHz lock in warm ambient the package zones can run away
(96 °C, hard SoC shutdown — silent below the OS); at 2200 MHz + SW
throttle the same load shape rode 92–96 °C stably with zero errors.

**The decode cost of the lower lock is zero, measured** — same node, same
bench, drafter on:

| | 2400 MHz lock | 2200 MHz lock |
|---|---|---|
| coding c1 | 37.8 | 38.3 |
| coding c4 aggregate | 82.6 | 83.2 |
| coding c8 aggregate | 123.6 | 137.4 |
| prose c1 | 22.0 | 23.7 |

Every figure is within (or above) run-to-run variance: decode is
bandwidth-bound, so the SM clock ceiling never binds there — only prefill
pays for clocks, and prefill is precisely the phase that overheats the
package.

**The prefill tax, measured** (same node, engine restarted fresh under each
lock, identical warmup, 75 s of unique ~4.6K-token prompts at c6 with
`max_tokens=16` — near-pure prefill):

| lock | sustained prefill | peak GPU / power during burst |
|---|---|---|
| 2400 MHz | 1,718 tok/s | 85 °C / 94 W (throttle already shaving: clocks dipped to 2288) |
| 2200 MHz | 1,432 tok/s | 83 °C / 80 W, clocks steady |

That is a **~17% prefill cost** — notably more than the 8% the clock ratio
alone predicts, so the graphics lock appears to drag other domains (uncore/
boost) with it on this platform. The trade in practice: prefill is the only
phase that pays, decode is free, and the 2400-lock burst was already
brushing the throttle at 94 W from a cold start — sustained 2400 prefill is
exactly the profile that produced the hard SoC shutdown. Single 75 s run
per point; treat as ±5%. 2200 is therefore the recommended serving lock for sustained
mixed traffic on this platform unless your ambient is generously cooled.
If you serve heavy-prefill workloads on a GB10, watch the ACPI package
zones (`/sys/class/thermal`), not `nvidia-smi`'s GPU temperature — the
firmware trip follows the zones, which run ~10 °C hotter.

## Prefill kernel pass (2026-07-28, dgx2 @ 2200 MHz)

Focus: close the large-M gap without regressing single-stream or light
concurrency decode. Microbench = one MoE layer, CUDA-graph capture,
achieved weight-bandwidth GB/s vs 273 GB/s LPDDR roofline. JSON under
`results/prefill-*.json`.

### Baseline (prior custom kernel, fixed BM=16/BK=64/BN=64/w4/s2)

| M | custom | tuned stock Triton |
|---|---|---|
| 1 | 172 | 126 |
| 64 | 232 | 114 |
| 256 | 221 | 139 |
| 512 | 196 | 133 |
| 1024 | 139 | 115 |
| 2048 | 79 | 67 |
| 4096 | **41.5** | **43.7** |

Decode was already the strong leg. Prefill cliff starts past M≈512; at
M=4096 custom was *parity or slightly behind* stock Triton (compute-bound
through dequant + bf16 tensor cores).

### What was tried (and measured)

1. **Fat tiles for prefill (BM=64/BK=128/w8, or BN=128 layout).**
   Hypothesis: fewer CTAs re-dequant the same expert weights.
   Result: **register spill.** M=4096 41.5 → 24.9 GB/s (fat BM/BK); BN=128
   layout 41 → **9.4 GB/s** and also cost 12% at M=1. Dead end. JSON:
   `prefill-tiles-v1-2200.json`, `prefill-bn128-2200.json`.

2. **Group-scale load rewrite** (one scale vector per K-group via
   `static_range`). Failed to compile cleanly in Triton on this stack;
   reverted. Not the lever.

3. **bf16 dequant output** (mul in fp32, narrow before `tl.dot` instead of
   casting at the MMA site). Kept — decode-neutral, cleaner TC feed.

4. **M-dependent tile select, matching the tuned stock prefill shape.**
   Stock `configs/...int4_w4a16.json` at M≥1024 uses **BM=64, BK=32, w4, s2**
   (small K when M is large — opposite of the dead-end above). Sweep
   (`bench/sweep_prefill_tiles.py` → `prefill-tile-sweep-2200.json`):

   | M | best tile | GB/s | decode tile (16/64) |
   |---|---|---|---|
   | 256 | 16/64 s2 | 217 | 217 |
   | 512 | **32/32 s3** | **201** | 188 |
   | 1024 | **64/32 s3** | **166** | 131 |
   | 2048 | **64/32 s3** | **94** | 76 |
   | 4096 | **64/32 s2–3** | **61** | 40 |

### Landed selector v3 (`kernels/w4a16_moe.py::_select_tiles`)

Wide sweep through M=8192 (`prefill-tile-sweep-wide-2200.json`) refined the
table — BM=128/BK=32 is the mid-prefill winner; BM=64/BK=64 wins the long
tail; BM=8 trims single-stream pad:

```
M ≤ 8     →  BM=8,   BK=64, w4 s2   # single-stream decode
M ≤ 128   →  BM=16,  BK=64, w4 s2   # light concurrency
M ≤ 256   →  BM=16,  BK=64, w4 s3
M ≤ 512   →  BM=32,  BK=64, w4 s3
M ≤ 1536  →  BM=64,  BK=64, w4 s3
M ≤ 3072  →  BM=128, BK=32, w4 s3   # +23% at M=2048 vs 64/32
M ≥ 4096  →  BM=64,  BK=64, w4 s3
```

`BLOCK_N=64` stays fixed (repack layout).

### Final microbench @ 2200 MHz (`results/prefill-v3-2200.json`)

| M | custom v3 | tuned stock Triton | vs morning baseline custom |
|---|---|---|---|
| 1 | **182** | 127 | **+6%** |
| 4 | **211** | 132 | +0% |
| 16 | **238** | 121 | +4% |
| 64 | **239** | 109 | +3% |
| 256 | **228** | 137 | +3% |
| 512 | **209** | 132 | +7% |
| 1024 | **171** | 115 | **+23%** |
| 2048 | **116** | 67 | **+46%** |
| 4096 | **66** | 44 | **+58%** |
| 8192 | **36** | 24 | **+50%** vs triton |

Custom wins every M. Prefill band cleared the +20% bar in the microbench;
long-context e2e confirms it (below).

### Lesson

On this GPU, **growing BK with BM spills; growing BM while *shrinking* BK
works** for some bands, and **BM=128 with BK=32** is the mid-prefill sweet
spot. Blindly "make all tiles bigger" is how you lose 40% overnight.

## End-to-end TPS A/B (2026-07-28, dgx2 @ 2200 MHz)

Same box, same weights, same `laguna-spark-boot` posture path, DFlash-NVFP4
@7, util 0.80, ctx 262K. Only the image/kernel changed:

- **baseline:** `vllm-laguna-w4a16:v0.1` (prior fixed-tile custom kernel)
- **candidate:** `vllm-laguna-w4a16:v0.2-prefill` (M-aware selector v3)

Harness: `bench/e2e_tps.py` — **cold** prompts only (unique nonce head, never
probed before the timed call, so prefix cache cannot inflate prefill). Coding
profile, 300 completion tokens, median of 2 runs. JSON:
`results/e2e-v01-cold-*.json`, `results/e2e-v02-cold-*.json`.

DFlash health both runs: ~2.1 accepted tokens / draft (healthy).

### Single-stream cold prefill + decode

| prompt (actual ≈) | v0.1 prefill tok/s | **v0.2 prefill** | Δ | v0.1 decode | **v0.2 decode** | Δ | v0.1 TTFT | **v0.2 TTFT** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~110 (short) | 290 | 283 | −2% | 25.8 | **28.5** | **+10%** | 0.38s | 0.39s |
| ~40.6K | 1615 | **2197** | **+36%** | 27.4 | **29.3** | +7% | 25.2s | **18.5s** |
| ~81.2K | 1427 | **1881** | **+32%** | 27.0 | 25.7 | −5% | 56.9s | **43.2s** |
| ~142K | 1243 | **1558** | **+25%** | 22.1 | **31.5** | **+42%** | 114.3s | **91.2s** |

Long-context prefill clears the **+20%** target at 40k / 80k / 140k. TTFT
drops ~20–27% in lockstep. Short-prompt decode is +10% (below 20% — decode
was already near the LPDDR ceiling in the MoE kernel; the residual is
attention / draft / sampler).

### Concurrency (c4)

| workload | v0.1 | **v0.2** | Δ |
|---|---:|---:|---:|
| short, agg decode tok/s | 58.2 | **63.5** | **+9%** |
| short, per-stream decode med | 15.5 | **17.4** | **+12%** |
| ~40k cold, per-stream prefill med | 602 | **815** | **+35%** |
| ~40k cold, agg decode tok/s | 10.3 | **12.8** | **+24%** |

### Image

```
docker build -t vllm-laguna-w4a16:v0.2-prefill \
  --build-arg BASE_IMAGE=vllm-laguna-nvfp4:v0.25.1 .
# launch via posture:
LAGUNA_IMAGE=vllm-laguna-w4a16:v0.2-prefill laguna-spark-boot laguna
```

Verified in-container: `_select_tiles(2048) == (128, 32, 4, 3)`,
`_select_tiles(1) == (8, 64, 4, 2)`, log line
`Using CompressedTensorsWNA16MoEMethod`, load 69.71 GiB.

## Unified memory: the allocator ratchet (2026-07-29)

On GB10 the GPU pool **is** system RAM, which turns an ordinary PyTorch
behaviour into an operational hazard. Serving Laguna, the engine's footprint
climbs for the first few minutes of traffic and then stops — but it climbs to
well past what you asked for, and every byte of that comes out of the host.

Two facts explain it, and the second is the one that bites:

1. `--gpu-memory-utilization` **does not cap the process.** vLLM never calls
   `torch.cuda.set_per_process_memory_fraction` (grep the source). The flag
   only sizes the KV cache at startup.
2. The caching allocator keeps every transient prefill activation it ever
   allocates. Nothing returns it while the process lives.

So the steady-state footprint is *reservation + peak transient*, and on a
discrete GPU you'd never notice — it would sit in spare VRAM. Here it comes
out of the machine's RAM.

**It is not a leak.** Measured on one GB10 (util 0.80, 6 identical bursts of
6 concurrent unique ~4.6K-token prompts, sampling per-process device memory
via `nvidia-smi --query-compute-apps`):

- growth **saturates** — cycles 5 and 6 landed on the same byte (105,126 MiB)
- stopping the engine returns **100%** of it (host free went back above its
  pre-launch value)
- process RSS never moved, so nothing leaks host-side either

### What to set, and what not to bother trying

`--max-num-batched-tokens` sizes the transient, so it is the lever that works.
Halving it halves the growth **and enlarges the KV cache** — vLLM derives KV
size from the profiled activation peak, and a smaller batch profiles smaller:

| | mnbt 8192 | **mnbt 4096** |
|---|---|---|
| growth over the run | +10.9 GiB | **+5.6 GiB** |
| KV cache | 447,778 tok (1.71x @262K) | **541,813 tok (2.07x)** |
| host free, saturated | 9.8 GiB | **12.7 GiB** |
| sustained prefill | 1846 tok/s | **1892 tok/s** |

There is no measured downside — throughput is unchanged and capacity improves,
which is why `serve.sh` defaults to 4096.

Measured and rejected, so you don't repeat them:

| attempt | result |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **worse** — ended 2.5 GiB higher, 5.8 GiB host free |
| `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` | **worse** — 0.6 GiB higher than stock |
| `--gpu-memory-utilization 0.75` | 2.2 GiB of headroom for **28% of the KV cache** (360,271 tok, 1.37x) — bad trade |

### If you need a hard guarantee

Everything above is budgeting: the footprint is predictable, so leave room for
it. If you must *guarantee* the engine cannot take the machine down (unattended
boxes, or a node shared with other containers), the missing piece is a real
cap. `integration/` documents a five-line, env-gated patch adding the
`set_per_process_memory_fraction` call vLLM omits, which converts an overshoot
into a failed request instead of a livelocked host.

# TASK 02 — GB10 config tuner for the int4-w4a16 fused-MoE path

Write `bench/tune_configs.py`. It runs inside the vLLM container on a GB10 and
produces `configs/E=256,N=1024,device_name=NVIDIA_GB10,dtype=int4_w4a16.json`
in exactly the format vLLM's `fused_moe` config loader expects (top-level keys
are M-bucket strings, values are config dicts).

## Baseline to beat (measured, this repo, `results/baseline.json`)

| M | triton default | note |
|---|---|---|
| 1 | **CRASH** | default config illegal for group 32: moe_wna16 CUDA path requires `BLOCK_SIZE_K // group_size ∈ {1,2,4,8}` |
| 4 | 102.5 GB/s | |
| 16 | 111.3 GB/s | |
| 64 | 102.5 GB/s | |
| 256 | 122.0 GB/s | |
| 1024 | 104.0 GB/s | |
| 4096 | 44.2 GB/s | |

Roofline 273 GB/s. Acceptance: **no crashes at any M (1 included)** and p50 ≥
170 GB/s for M ∈ {1, 4, 16, 64, 256} (decode band), no regression at 1024/4096.
If 170 turns out unreachable by config choice alone, deliver the best
configuration found and write the measured ceiling into the json's sidecar
notes file — that becomes the justification for a custom kernel (TASK 03).

## How

1. Study `reference/fused_moe.py`: `get_moe_configs`, `get_default_config`,
   `try_get_optimal_moe_config`, and `should_moe_wna16_use_cuda` — understand
   (a) how a config json is looked up and keyed by M, (b) which knobs exist
   (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`), and (c)
   when the CUDA `moe_wna16` kernel is used instead of triton (small M) and
   which of the knobs it consumes (its `BLOCK_SIZE_K//group ∈ {1,2,4,8}`
   constraint must be respected in configs for those M buckets).
2. The tuner: for each M in `1,2,4,8,16,24,32,48,64,96,128,256,512,1024,2048,4096`,
   sweep a sensible candidate grid, reusing this repo's `bench_moe.py`
   machinery (import it) for weight/routing construction and CUDA-event timing.
   Inject candidates by monkeypatching the config lookup
   (`fused_moe.get_moe_configs` / `try_get_optimal_moe_config`) — do NOT edit
   installed vLLM files.
3. Invalid candidates (triton compile failure, CUDA-path constraint, wrong
   results) are caught, recorded as invalid, and skipped. Correctness spot-check
   winners against the reference at M=16.
4. Emit the json + a `configs/NOTES-gb10.md` with the winning table (M, config,
   GB/s, % roofline).
5. Support `--quick` (coarser grid) and `--M` override. Print progress lines;
   this will run unattended for a while.

Style rules as TASK-01: standalone, argparse, no site-specific strings.

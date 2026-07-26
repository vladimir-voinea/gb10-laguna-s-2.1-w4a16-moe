# TASK 01 — W4A16 grouped-MoE microbenchmark

Write `bench/bench_moe.py`: a single-file, standalone correctness + performance
benchmark for W4A16 mixture-of-experts GEMM, runnable inside a vLLM 0.25.1
container (`--entrypoint python3`, repo bind-mounted at `/repo`). No pip
installs — only torch, triton and the installed vllm may be imported.

## Quantization contract (matches a compressed-tensors `pack-quantized` MoE)

- Weights: INT4, **symmetric** (no zero-points), **group_size 32** along K,
  scales bf16.
- Activations: bf16, unquantized.
- Only the routed experts are quantized.

## Model shapes (default preset `laguna117b`, one MoE layer)

- E = 256 experts, top_k = 10, renormalized softmax routing (random logits ok)
- hidden K = 3072, intermediate N = 1024
- w13 (gate+up fused): logical [E, 2N=2048, K=3072] int4
- w2 (down):           logical [E, K=3072, N=1024] int4
- SiLU-and-mul between the two GEMMs (standard fused-MoE dataflow)
- dtype of activations/outputs: bf16

Also accept `--E/--K/--N/--topk/--group` overrides.

## Backends (a `--backend` flag; `all` runs each)

1. `reference` — pure PyTorch: dequantize int4→bf16 (per-group scale), loop
   experts with plain `torch.matmul`. Slow is fine; this is the correctness
   anchor.
2. `triton` — vLLM's own path: `vllm.model_executor.layers.fused_moe.fused_moe
   .fused_experts(..., use_int4_w4a16=True, ...)`. Read
   `reference/fused_moe.py` (vendored copy of the installed source) to get the
   EXACT packing/layout it expects: how qweights are packed (nibble order,
   shape), scale tensor shapes, `block_shape` argument, topk tensors. Do not
   guess — derive from the source, and make the bench pack its synthetic
   weights into precisely that layout.
3. `marlin` — best-effort: vLLM's fused Marlin MoE with
   `gptq_marlin_moe_repack`-style packing. If wiring this inside one file gets
   gnarly, make it cleanly skippable (`--backend marlin` prints SKIPPED with
   the reason) — the Marlin end-to-end anchor already exists (10.8 tok/s).

## Measurement

- M sweep (tokens per forward): default `1,4,16,64,256,1024,4096`.
- Per (backend, M): warmup 10 iters, then ≥50 timed iters with CUDA events;
  report p50 latency ms.
- Report **achieved weight-bandwidth**: bytes of quantized weights + scales the
  call must read (count only experts activated by the routing for that M —
  with random top-10 routing over 256 experts, count the UNIQUE experts hit,
  which for M≥64 is effectively all E) divided by p50 time. Print GB/s and %
  of a `--roofline` (default 273 GB/s).
- Correctness: every backend's output vs `reference` on the same inputs at
  M=16 — report max abs err and rel Frobenius err; FAIL loudly above
  rtol 3e-2 (bf16 + int4 tolerances).
- `--json results/<name>.json` dump with all numbers + shapes + device name.

## Output

A compact aligned table to stdout:

```
backend   M     p50 ms   GB/s    %roofline   ok
triton    1     …        …       …           ✓
```

## Style

Plain, readable, no framework. Argparse. Type hints. A `--seed`. Deterministic
inputs. Exit non-zero on correctness failure. Keep every identifier generic
(no site-specific names, hosts, or paths).

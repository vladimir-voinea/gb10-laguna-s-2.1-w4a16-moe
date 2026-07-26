# Serving a WNA16 (int4, group-32) MoE checkpoint fast on GB10

Three pieces, all opt-in:

## 1. Device-tuned Triton configs

Copy (or bind-mount) the tuned config into the installed vLLM:

```bash
# copy at container start (a per-file -v bind can silently mangle this
# comma-and-equals-heavy filename; copying from a mounted dir is robust):
docker run … -v $PWD/configs:/tuned:ro --entrypoint bash <image> -c '
  cp /tuned/E*.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/ &&
  exec vllm serve …'
```

Without this the Triton path uses default tile sizes (and, before tuning, the
small-M default config was outright **illegal** for group-32 on the moe_wna16
CUDA sub-path: `BLOCK_SIZE_K // group_size must be one of [1,2,4,8]` — i.e.
single-token decode crashed, which is why Marlin looked like the only option).

## 2. Route WNA16 MoE to Triton

```bash
python3 integration/apply.py     # inside the image/venv, once
```

then serve with:

```bash
vllm serve <model> --moe-backend triton …
```

Stock vLLM hard-prefers Marlin for WNA16 MoE on CUDA; the patch makes an
explicit `--moe-backend triton` win. No flag → stock behavior, byte-identical.

## 3. (optional) The custom kernel

`kernels/w4a16_moe.py` goes further than the tuned stock kernel (contiguous
N-tile-major streaming layout + fused SiLU). Wiring it into serving requires a
custom quant-method class; see the top-level README for status. The microbench
(`bench/bench_moe.py --backend custom`) is the reference for its numbers.

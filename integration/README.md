# Serving a WNA16 (int4, group-32) MoE checkpoint fast on GB10

Three pieces, all opt-in:

## 1. Device-tuned Triton configs

Copy (or bind-mount) the tuned config into the installed vLLM:

```bash
CFG='E=256,N=1024,device_name=NVIDIA_GB10,dtype=int4_w4a16.json'
docker run … \
  -v $PWD/configs/$CFG:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/$CFG:ro \
  …
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

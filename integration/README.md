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
N-tile-major streaming layout + fused SiLU). Enable it with:

```bash
python3 integration/apply.py --custom     # inside the image/venv, once
export GB10_W4A16_CUSTOM=1
vllm serve <model> --moe-backend triton …
```

What `--custom` does:

1. Applies the backend patch from §2 (if not already applied).
2. Copies `kernels/w4a16_moe.py` → `site-packages/gb10_w4a16_moe.py` (site-packages
   taken from the installed `vllm` module path).
3. Patches installed `compressed_tensors_moe_wna16.py` so that with
   `GB10_W4A16_CUSTOM=1`:
   - `process_weights_after_loading` repacks post-load w13/w2 into the
     N-tile-major layout via `gb10_w4a16_moe.repack_weights` (stock 3-D
     packed qweights are freed to avoid ~2× expert VRAM).
   - `apply()` routes to `gb10_w4a16_moe.fused_experts_w4a16` with the topk
     tensors it already has. Env unset or repack absent → stock path.

Leave `GB10_W4A16_CUSTOM=1` for the whole process lifetime when using custom
(weights are freed after repack). Revert with `python3 integration/apply.py --revert`.

Layout note: post-load weights already match the microbench packs
`(E, N_out, K_in//2)` — `create_weights` stores transposed int32 and
`process_weights_after_loading` does the only needed transpose; repack takes
that form directly.

The microbench (`bench/bench_moe.py --backend custom`) is the reference for
kernel numbers.

# Fresh DGX Spark → fast W4A16 Laguna, step by step

Everything below is pinned to the exact versions this repo was validated with.
Substituting newer revisions may work but is not what was measured.

## 0. Pinned versions

| component | pin |
|---|---|
| this repo | tag **`v0.2`** — includes the repack scales fix (v0.1 loses ~7 GiB of KV pool). Kernels unchanged since `0e4b688`. |
| vLLM | 0.25.1 (inside the base image) |
| base docker image | any vLLM 0.25.1 image with triton + sm_121a; validated against a local build (`vllm-laguna-nvfp4:v0.25.1`, image id `6130ea047629`); public equivalent `vllm/vllm-openai:v0.25.1` |
| target checkpoint | `poolside/Laguna-S-2.1-INT4` @ revision **`67dbeda456e68139f281c40831f9d12049d8fc11`** |
| draft (speculative) | `poolside/Laguna-S-2.1-DFlash-NVFP4` @ revision **`723794750422b3efbf3a7b3af76dffb4ba035943`** — yes, the **NVFP4** draft: it is trained against the same BF16 base and accepts ~2.25 tokens/window vs the INT4 target. The `DFlash-INT4` repo draft accepts **0%** (older weight snapshot); do not use it. |
| tuned config | `configs/E=256,N=1024,device_name=NVIDIA_GB10,dtype=int4_w4a16.json` (in this repo) |

## 1. Get the weights

```bash
hf download poolside/Laguna-S-2.1-INT4 \
  --revision 67dbeda456e68139f281c40831f9d12049d8fc11 \
  --local-dir ~/models/Laguna-S-2.1-INT4          # ~67 GB
hf download poolside/Laguna-S-2.1-DFlash-NVFP4 \
  --revision 723794750422b3efbf3a7b3af76dffb4ba035943 \
  --local-dir ~/models/Laguna-S-2.1-DFlash-NVFP4  # ~2.1 GB
```

## 2. Build the image

```bash
git clone <this-repo> && cd gb10-laguna-s-2.1-w4a16-moe && git checkout v0.2
docker build -t vllm-laguna-w4a16:v0.2 \
  --build-arg BASE_IMAGE=<your vllm 0.25.1 image> .
```

The build bakes in the custom kernel, the tuned GB10 configs, the vLLM patch
(`apply.py --custom`), and sets `GB10_W4A16_CUSTOM=1`. The build fails loudly if
the patch anchors don't match your vLLM — that is deliberate (version drift
guard).

## 3. Serve

`./serve.sh` in the repo root is this command, parameterized — the exact
production configuration (parsers, thinking defaults, generation config).
The expanded form:

```bash
docker run -d --name laguna-w4a16 --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 \
  -v ~/models/Laguna-S-2.1-INT4:/model:ro \
  -v ~/models/Laguna-S-2.1-DFlash-NVFP4:/dflash:ro \
  vllm-laguna-w4a16:v0.2 \
  --model /model --served-model-name laguna-w4a16 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 --max-model-len 262144 --max-num-seqs 32 \
  --moe-backend triton \
  --speculative-config '{"model":"/dflash","num_speculative_tokens":7,"method":"dflash"}'
```

Notes that matter:
- `--moe-backend triton` is required — it is the opt-in the baked patch honors;
  without it vLLM silently uses Marlin.
- `--max-num-seqs` must stay ≤ 32 with the DFlash drafter (crashes at the
  vLLM default of 256).
- If you run under Tegra-MPS, add your MPS mounts and keep
  `VLLM_USE_FLASHINFER_SAMPLER=0` — FlashInfer sampling deadlocks under MPS
  on GB10.
- **GPU memory IS system memory on GB10.** Concurrent long prefills can
  spike allocations past the `--gpu-memory-utilization` reservation, and the
  failure mode is not a clean CUDA OOM — it drains the whole box to zero and
  livelocks it (kernel `NV_ERR_NO_MEMORY`, then nothing answers until a
  power cycle). Leave headroom (0.80 is what we run), and if you use
  earlyoom, trigger it on memory alone (`-s 100,100`): GPU-pinned pages
  never swap, so a swap-gated config will watch the box die at 0.00% free
  with swap untouched.

## 4. Verify (all three, in order)

```bash
# 1. the right code path (must say WNA16MoEMethod, not Marlin):
docker logs laguna-w4a16 2>&1 | grep "Using Compressed"
#    -> Using CompressedTensorsWNA16MoEMethod

# 2. the drafter is alive (accepted must be climbing, ~2.2/draft):
curl -s localhost:8000/metrics | grep -E "spec_decode_num_(drafts|accepted_tokens)_total"

# 3. throughput (this repo):
python3 bench/e2e_tps.py --url http://localhost:8000/v1 --model laguna-w4a16 \
  --runs 2 --max-tokens 800
```

Expected on a GB10 (single stream, 800-token decode): **~35–40 tok/s coding,
~22 tok/s prose**; c8 aggregate ~120 tok/s. Also check capacity — `Model loading took` should equal your checkpoint
plus the draft (~69.7 GiB for this pair). If it is several GiB higher you are
on v0.1, which keeps the stock scales alongside the repacked ones. If you see ~19 spec-off-level
numbers, check #2 (drafter). If you see ~11, the drafter is accepting 0% —
wrong draft/target pairing.

## Troubleshooting

| symptom | cause |
|---|---|
| build succeeds but serving still uses Marlin | `--moe-backend triton` missing from the serve command (the baked patch is opt-in by design) |
| `BLOCK_SIZE_K // group_size must be one of [1,2,4,8]` at M=1 | tuned config json missing — the vLLM default config is illegal for group-32 small-M |
| `Using CompressedTensorsWNA16MarlinMoEMethod` in logs | `--moe-backend triton` missing |
| tok/s ≈ half of expected, acceptance = 0 | wrong draft (see pin table — use the NVFP4 draft) |
| docker `-v` of a single config json mounts a mangled filename | copy from a mounted dir instead (or use this image, which bakes them) |



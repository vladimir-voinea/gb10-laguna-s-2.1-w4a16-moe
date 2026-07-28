#!/usr/bin/env bash
# Serve Laguna-S-2.1-INT4 on a GB10 exactly as we run it in production —
# same flags, same generation config, same parsers. Override any UPPERCASE
# var via the environment.
#
#   ./serve.sh                          # after: docker build -t vllm-laguna-w4a16:v0.2 .
#   MODEL_DIR=/data/int4 PORT=8001 ./serve.sh
set -euo pipefail

IMAGE="${IMAGE:-vllm-laguna-w4a16:v0.2}"
NAME="${NAME:-laguna-w4a16}"
PORT="${PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Laguna-S-2.1-INT4}"
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/Laguna-S-2.1-DFlash-NVFP4}"
SERVED_NAME="${SERVED_NAME:-laguna-s-2.1}"

# GB10 memory is UNIFIED — the GPU pool is system RAM. 0.80 leaves ~19 GB of
# headroom; concurrent long prefills can spike past the reservation, and the
# failure mode is the whole box livelocking, not a clean CUDA OOM. Don't get
# greedy here (see RESULTS.md).
UTIL="${UTIL:-0.80}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-32}"          # DFlash crashes vLLM at the default 256

# Sampling defaults the model was tuned for (harnesses can still override
# per-request).
GENCFG='{"temperature":0.7,"top_p":0.95,"repetition_penalty":1.0}'

# Uncomment if you run under Tegra-MPS (and keep FI sampler off — FlashInfer
# sampling deadlocks under MPS on GB10):
#MPS_ARGS=(-v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps)
MPS_ARGS=()

docker run -d --name "$NAME" --restart no --gpus all --ipc=host --shm-size 16g \
  -p "$PORT:8000" \
  "${MPS_ARGS[@]}" \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL_DIR:/model:ro" -v "$DFLASH_DIR:/dflash:ro" \
  "$IMAGE" \
  --model /model --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --enable-prefix-caching --max-num-batched-tokens 8192 \
  --no-async-scheduling \
  --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}' \
  --override-generation-config "$GENCFG" \
  --reasoning-parser poolside_v1 --tool-call-parser poolside_v1 --enable-auto-tool-choice \
  --moe-backend triton \
  --speculative-config '{"model":"/dflash","num_speculative_tokens":7,"method":"dflash"}' \
  --gpu-memory-utilization "$UTIL" --max-model-len "$CTX" --max-num-seqs "$SEQS" \
  --limit-mm-per-prompt '{"image":256}'

echo "serving $SERVED_NAME on :$PORT  (util=$UTIL ctx=$CTX seqs=$SEQS, drafter=DFlash-NVFP4)"
echo "verify:  docker logs $NAME 2>&1 | grep 'Using Compressed'   # must say WNA16MoEMethod"

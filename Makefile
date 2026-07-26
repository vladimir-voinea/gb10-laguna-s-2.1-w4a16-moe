# BENCH_HOST: a host with a GB10 and the vLLM image (set in your environment).
# IMAGE: the vLLM container image to run the bench inside.
BENCH_HOST ?= $(error set BENCH_HOST to your GB10 host)
IMAGE ?= vllm-laguna-nvfp4:v0.25.1
REMOTE_DIR ?= /tmp/gb10-w4a16-moe
M ?= 1,4,16,64,256,1024,4096
ARGS ?=

sync:
	rsync -a --delete --exclude .git --exclude results ./ $(BENCH_HOST):$(REMOTE_DIR)/

bench: sync
	ssh $(BENCH_HOST) 'docker run --rm --gpus all -v $(REMOTE_DIR):/repo \
	  --entrypoint python3 $(IMAGE) /repo/bench/bench_moe.py \
	  --M $(M) --json /repo/results/latest.json $(ARGS)'
	mkdir -p results && rsync -a $(BENCH_HOST):$(REMOTE_DIR)/results/ results/

.PHONY: sync bench

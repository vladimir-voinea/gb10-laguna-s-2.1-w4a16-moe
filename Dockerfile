# Bakes this repo's W4A16 path into a vLLM image:
#   - kernels/w4a16_moe.py            -> site-packages/gb10_w4a16_moe.py
#   - integration/apply.py --custom   -> patches WNA16 MoE method + backend pick
#   - configs/*.json                  -> vLLM fused_moe tuned configs (GB10)
#   - ENV GB10_W4A16_CUSTOM=1         -> custom kernel active for WNA16 MoE
#
# The base must be a vLLM >= 0.25.x image with triton and sm_121a support.
# Serve with:  --moe-backend triton   (plus your usual flags; see SETUP.md)
ARG BASE_IMAGE=vllm/vllm-openai:v0.25.1
FROM ${BASE_IMAGE}

COPY kernels/w4a16_moe.py /opt/gb10-w4a16/kernels/w4a16_moe.py
COPY integration/apply.py /opt/gb10-w4a16/integration/apply.py
COPY configs/*.json /opt/gb10-w4a16/configs/

RUN set -e; \
    SP=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)"); \
    cp /opt/gb10-w4a16/configs/E*.json \
       "$SP/vllm/model_executor/layers/fused_moe/configs/"; \
    python3 /opt/gb10-w4a16/integration/apply.py --custom; \
    python3 -c "import gb10_w4a16_moe; print('kernel import OK')"

ENV GB10_W4A16_CUSTOM=1

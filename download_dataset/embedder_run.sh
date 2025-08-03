#!/bin/bash
export NGC_API_KEY=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
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"

# Log into NVIDIA Container Registry
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -e NIM_TRITON_PERFORMANCE_MODE=throughput \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    -p 8001:8001 \
    nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
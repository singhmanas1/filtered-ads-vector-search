#!/bin/bash
export NGC_API_KEY=nvapi-xxx
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
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
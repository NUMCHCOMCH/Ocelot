#!/bin/bash

# vLLM 서버 실행 스크립트
# Ocelot-Ko-self-instruction-10.8B-v1.0 모델 로드

echo "vLLM 서버를 시작합니다..."
echo "모델: cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0"
echo "GPU: 0,1 사용"
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --max-seq-len-to-capture 1024 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --disable-log-requests

echo ""
echo "vLLM 서버가 종료되었습니다."
#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-m}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-69632}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-69632}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
LOG_FILE="${LOG_FILE:-/tmp/qwen35-vllm-repro.log}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"

if [[ -z "${ADDITIONAL_CONFIG}" ]]; then
    ADDITIONAL_CONFIG='{"enable_sleep_mode_extra_cleanup":true}'
fi

export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

args=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --enable-sleep-mode
    --enable-prefix-caching
    --enable-chunked-prefill
    --no-disable-hybrid-kv-cache-manager
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config "${ADDITIONAL_CONFIG}"
)

if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
    args+=(--async-scheduling)
fi

echo "Starting Qwen3.5 vLLM repro server"
echo "  model=${MODEL_PATH}"
echo "  endpoint=http://${HOST}:${PORT}"
echo "  tp=${TP_SIZE}, async_scheduling=${ASYNC_SCHEDULING}"
echo "  log=${LOG_FILE}"

"${args[@]}" 2>&1 | tee "${LOG_FILE}"

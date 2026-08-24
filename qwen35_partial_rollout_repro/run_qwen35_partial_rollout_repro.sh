#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-m}"
CYCLES="${CYCLES:-50}"
INFLIGHT_DELAY_SECONDS="${INFLIGHT_DELAY_SECONDS:-2}"
INFLIGHT_MAX_TOKENS="${INFLIGHT_MAX_TOKENS:-65536}"
PREFILL_REQUEST_FILE="${PREFILL_REQUEST_FILE:-${SCRIPT_DIR}/qwen35_prefill_request.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/qwen35-partial-rollout-repro}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${PREFILL_REQUEST_FILE}" ]]; then
    echo "Missing prefill request file: ${PREFILL_REQUEST_FILE}" >&2
    exit 1
fi

echo "Checking ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health" >/dev/null

for i in $(seq 1 "${CYCLES}"); do
    echo "cycle=${i}/${CYCLES}: start an in-flight decode request"

    curl -sS "${BASE_URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Write a very long Python program and keep explaining it.\",\"max_tokens\":${INFLIGHT_MAX_TOKENS},\"temperature\":0}" \
        >"${OUTPUT_DIR}/inflight-${i}.json" &
    request_pid=$!

    sleep "${INFLIGHT_DELAY_SECONDS}"

    echo "cycle=${i}: pause/abort unfinished requests"
    curl -fsS -X POST "${BASE_URL}/pause?mode=abort&clear_cache=true"
    wait "${request_pid}" || true

    echo "cycle=${i}: sleep level 1"
    curl -fsS -X POST "${BASE_URL}/sleep?level=1&mode=abort"

    echo "cycle=${i}: staged wake weights -> kv_cache"
    curl -fsS -X POST "${BASE_URL}/wake_up?tags=weights"
    curl -fsS -X POST "${BASE_URL}/wake_up?tags=kv_cache"
    curl -fsS -X POST "${BASE_URL}/resume"

    echo "cycle=${i}: run a fresh GDN prefill"
    if ! curl -fsS "${BASE_URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@${PREFILL_REQUEST_FILE}" \
        >"${OUTPUT_DIR}/prefill-${i}.json"; then
        echo "Reproduction failed at cycle=${i}; inspect the vLLM log and ${OUTPUT_DIR}." >&2
        exit 1
    fi

    if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
        echo "vLLM became unhealthy at cycle=${i}; inspect the vLLM log." >&2
        exit 1
    fi
done

echo "Completed ${CYCLES} cycles without reproducing the crash."
echo "Outputs: ${OUTPUT_DIR}"

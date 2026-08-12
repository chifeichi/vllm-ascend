#!/bin/bash

set -e

curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3",
    "prompt":"Please briefly introduce yourself.",
    "max_tokens":1,
    "temperature":0
  }' > warmup.json

PROMPT1="$(python3 - <<'PY'
text = "The following is a persistent context used to test cross-turn KV cache reuse. "
print(text * 1000 + "\nPlease summarize the context in detail.")
PY
)"

curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model qwen3 \
    --arg prompt "$PROMPT1" \
    '{model:$model,prompt:$prompt,max_tokens:1024,temperature:0}')" \
  > response1.json

jq '{prompt_tokens:.usage.prompt_tokens,output_tokens:.usage.completion_tokens,finish_reason:.choices[0].finish_reason,text:.choices[0].text}' response1.json

ANSWER1="$(jq -r '.choices[0].text' response1.json)"
PROMPT2="${PROMPT1}${ANSWER1}

Based on everything above, provide three additional conclusions."

curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model qwen3 \
    --arg prompt "$PROMPT2" \
    '{model:$model,prompt:$prompt,max_tokens:128,temperature:0}')" \
  > response2.json

jq '{prompt_tokens:.usage.prompt_tokens,output_tokens:.usage.completion_tokens,finish_reason:.choices[0].finish_reason,text:.choices[0].text}' response2.json

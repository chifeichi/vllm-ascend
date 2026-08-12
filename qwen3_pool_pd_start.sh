#!/bin/bash

MODEL=/path/to/Qwen3-30B-A3B
VLLM=/path/to/vllm
VLLM_ASCEND=/path/to/cache_pool_worktree/vllm-ascend
MOONCAKE_JSON=/path/to/mooncake.json

export PYTHONPATH=$VLLM_ASCEND:$VLLM:$PYTHONPATH
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=$MOONCAKE_JSON
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ASCEND_BUFFER_POOL=4:8
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

nohup mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000 > mooncake_master.log 2>&1 &
MASTER_PID=$!

nohup env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3 \
  --port 8001 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"MultiConnector",
    "kv_role":"kv_producer",
    "kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{"connectors":[
      {
        "kv_connector":"MooncakeConnectorV1",
        "kv_role":"kv_producer",
        "kv_port":"20001",
        "kv_connector_extra_config":{
          "prefill":{"dp_size":1,"tp_size":4},
          "decode":{"dp_size":1,"tp_size":4}
        }
      },
      {
        "kv_connector":"AscendStoreConnector",
        "kv_role":"kv_producer",
        "kv_connector_extra_config":{
          "lookup_rpc_port":"0",
          "backend":"mooncake",
          "load_async":false,
          "use_layerwise":false
        }
      }
    ]}
  }' > p.log 2>&1 &
P_PID=$!

nohup env ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3 \
  --port 8002 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"MultiConnector",
    "kv_role":"kv_consumer",
    "kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{"connectors":[
      {
        "kv_connector":"MooncakeConnectorV1",
        "kv_role":"kv_consumer",
        "kv_port":"20002",
        "kv_connector_extra_config":{
          "prefill":{"dp_size":1,"tp_size":4},
          "decode":{"dp_size":1,"tp_size":4}
        }
      },
      {
        "kv_connector":"AscendStoreConnector",
        "kv_role":"kv_consumer",
        "kv_connector_extra_config":{
          "lookup_rpc_port":"1",
          "backend":"mooncake",
          "consumer_is_to_put":true,
          "store_decode_kv":true,
          "consumer_is_to_load":false,
          "load_async":false,
          "use_layerwise":false
        }
      }
    ]}
  }' > d.log 2>&1 &
D_PID=$!

until curl -sf http://127.0.0.1:8001/health >/dev/null && curl -sf http://127.0.0.1:8002/health >/dev/null; do
  if ! kill -0 $P_PID 2>/dev/null || ! kill -0 $D_PID 2>/dev/null; then
    wait $P_PID $D_PID
    exit 1
  fi
  sleep 5
done

nohup python3 "$VLLM_ASCEND/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8001 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8002 > proxy.log 2>&1 &
PROXY_PID=$!

printf '%s\n' "$MASTER_PID" "$P_PID" "$D_PID" "$PROXY_PID" > qwen3_pool_pd.pids
echo "master_pid=$MASTER_PID p_pid=$P_PID d_pid=$D_PID proxy_pid=$PROXY_PID"

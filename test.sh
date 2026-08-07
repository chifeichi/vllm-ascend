grep '\[VLLM_ASCEND_PD_TRANSFER\]' <日志文件> \
  | sed -E 's/.*transfer_window_ms=([^ ]+).*mooncake_call_ms=([^ ]+).*chunks=([0-9]+).*/\3 \1 \2/' \
  | awk '{n[$1]++; w[$1]+=$2; m[$1]+=$3} END {for (k in n) printf "chunks=%s count=%d avg_window_ms=%.1f avg_mooncake_ms=%.1f\n",k,n[k],w[k]/n[k],m[k]/n[k]}' \
  | sort

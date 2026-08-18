grep -a "VLLM_ASCEND_PD_TRANSFER" 1.log | awk '
{
  pid=""; tp=""; port=""; wait=""; depth=""
  for (i=1; i<=NF; i++) {
    split($i, a, "=")
    if (a[1]=="pid") pid=a[2]
    else if (a[1]=="tp_rank") tp=a[2]
    else if (a[1]=="slowest_peer_port") port=a[2]
    else if (a[1]=="peer_queue_wait_max_ms") wait=a[2]+0
    else if (a[1]=="peer_queue_depth_at_enqueue") depth=a[2]+0
  }
  if (port != "" && port != "-1") {
    key=pid " " tp " " port
    count[key]++
    wait_sum[key]+=wait
    depth_sum[key]+=depth
    if (wait>wait_max[key]) wait_max[key]=wait
  }
}
END {
  for (key in count)
    printf "%s samples=%d avg_wait_ms=%.1f max_wait_ms=%.1f avg_depth=%.1f\n",
      key, count[key], wait_sum[key]/count[key],
      wait_max[key], depth_sum[key]/count[key]
}'
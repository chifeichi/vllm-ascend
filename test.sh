grep -a "VLLM_ASCEND_PD_TRANSFER" 1.log | awk '
{
  for (i=1; i<=NF; i++) {
    if ($i ~ /^peer_stats=/) {
      value=$i
      sub(/^peer_stats=/, "", value)
      n=split(value, peers, ",")
      for (j=1; j<=n; j++) {
        m=split(peers[j], p, ":")
        if (m == 5) {
          port=p[1]
          samples[port]++
          tasks[port]+=p[2]
          total_bytes[port]+=p[3]
          wait_sum[port]+=p[4]
          mooncake_sum[port]+=p[5]
        }
      }
    }
  }
}
END {
  for (port in samples) {
    mib=total_bytes[port]/1024/1024
    seconds=mooncake_sum[port]/1000
    throughput=0
    if (seconds > 0)
      throughput=mib/seconds
    printf "port=%s samples=%d tasks=%d avg_bytes_mb=%.2f avg_wait_ms=%.1f avg_mooncake_ms=%.1f effective_mb_s=%.1f\n",
      port, samples[port], tasks[port],
      mib/tasks[port],
      wait_sum[port]/samples[port],
      mooncake_sum[port]/tasks[port],
      throughput
  }
}'
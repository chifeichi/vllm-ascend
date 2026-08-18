grep -a "VLLM_ASCEND_PD_TRANSFER" 1.log | awk '
{
  value=-1
  for (i=1; i<=NF; i++) {
    if ($i ~ /^task_queue_wait_max_ms=/) {
      split($i, a, "=")
      value=a[2]+0
    }
  }
  if (value > max) {
    max=value
    line=$0
  }
}
END {
  print "max_task_queue_wait_ms=" max
  print line
}'
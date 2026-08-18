grep -a "VLLM_ASCEND_PD_TRANSFER" 1.log | grep -v "status=completed" | head -n 1

grep -a "VLLM_ASCEND_PD_TRANSFER" 1.log | awk '
{
  value=-1
  for (i=1; i<=NF; i++) {
    if ($i ~ /^transfer_window_ms=/) {
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
  print "max_transfer_window_ms=" max
  print line
}'
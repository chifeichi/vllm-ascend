grep -a "MooncakeConnector finish req not in reqs to process" <vllm日志文件> \
| sed -nE 's/.*request_id=([^[:space:].,]+).*/\1/p' \
| sort \
| uniq -c \
| awk '
NR == 1 { min=$1; max=$1 }
{
  requests++;
  warnings += $1;
  if ($1 < min) min=$1;
  if ($1 > max) max=$1
}
END {
  print "warnings=" warnings;
  print "unique_requests=" requests;
  print "avg_per_request=" warnings / requests;
  print "min_per_request=" min;
  print "max_per_request=" max;
}'
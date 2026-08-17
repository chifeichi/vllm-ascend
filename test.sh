grep -a "MooncakeConnector finish req not in reqs to process" <本次新日志文件> \
| sed -nE 's/.*worker_tp([0-9]+) pid=[0-9]+\).*request_id=([^, .]+).*/\2 \1/p' \
| sort -k1,1 -k2,2n \
| awk '
$1 != request_id {
  if (request_id != "") {
    print "request_id=" request_id, "warnings=" count, "tp_ranks=" ranks
    shown++
    if (shown >= 10) exit
  }
  request_id=$1
  count=0
  ranks=""
}
{
  count++
  ranks = ranks (ranks == "" ? "" : ",") $2
}
'
grep -a "MooncakeConnector finish req not in reqs to process" <日志文件> \
| sed -nE 's/.*request_id=([^[:space:].,]+).*/\1/p' \
| sort \
| uniq -c \
| awk '{count[$1]++} END {for (n in count) print "warnings_per_request=" n, "requests=" count[n]}' \
| sort -n
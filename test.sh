grep -a "MooncakeConnector finish req not in reqs to process" <本次新日志文件> \
| sed -nE 's/.*\(worker_tp([0-9]+) pid=([0-9]+)\).*/tp=\1 pid=\2/p' \
| sort \
| uniq -c \
| sort -k2,2n
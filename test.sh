awk '
/VERL_PD_D_SCHEDULER/ {
  rr=run=remote=sched=""
  for (i=1; i<=NF; i++) {
    split($i,a,"=")
    if (a[1]=="replica_rank") rr=a[2]
    if (a[1]=="running") run=a[2]
    if (a[1]=="waiting_remote_kv") remote=a[2]
    if (a[1]=="scheduled_reqs") sched=a[2]
  }
  n[rr]++
  run_sum[rr]+=run
  remote_sum[rr]+=remote
  sched_sum[rr]+=sched
  if (run==0) idle[rr]++
  if (run==0 && remote>0) idle_remote[rr]++
  if (run>run_max[rr]) run_max[rr]=run
}
END {
  for (rr in n)
    printf "replica=%s samples=%d avg_running=%.2f max_running=%d avg_scheduled=%.2f avg_remote_wait=%.2f idle_ratio=%.3f idle_remote_ratio=%.3f\n",
      rr,n[rr],run_sum[rr]/n[rr],run_max[rr],sched_sum[rr]/n[rr],
      remote_sum[rr]/n[rr],idle[rr]/n[rr],idle_remote[rr]/n[rr]
}' <日志文件>
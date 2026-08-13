grep -a "VERL_PD_CACHE_FINISH" 1.log |
awk '
{
  delete v
  for (i=1; i<=NF; i++) {
    split($i,a,"=")
    v[a[1]]=a[2]
  }
  n++
  total += v["total_tokens"]
  retained += v["retained_hit_tokens_after_free"]
  if (v["retained_hit_tokens_after_free"] == 0) zero++
  if (v["hash_entries_after"] < v["hash_entries_before"]) shrunk++
}
END {
  print "samples=" n
  print "avg_total_tokens=" total/n
  print "avg_retained_after_free=" retained/n
  print "retained_ratio=" 100*retained/total "%"
  print "zero_retained_count=" zero+0
  print "hash_entries_shrunk_count=" shrunk+0
}'
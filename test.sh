grep "VERL_PD_CACHE_SCHED" 1.log |
awk '
{
  delete v
  for (i=1; i<=NF; i++) {
    split($i,a,"=")
    v[a[1]]=a[2]
  }

  n++
  session=v["session_id"]

  if (session in last_first_hash && last_first_hash[session] != v["first_hash"])
    first_hash_changed++

  if (session in last_salt && last_salt[session] != v["cache_salt"])
    cache_salt_changed++

  if (session in last_lora && last_lora[session] != v["lora_id"])
    lora_changed++

  if (session in last_mm && last_mm[session] != v["mm_features"])
    mm_changed++

  if (v["skip_prefix_read"] != "False")
    skip_prefix_read++

  last_first_hash[session]=v["first_hash"]
  last_salt[session]=v["cache_salt"]
  last_lora[session]=v["lora_id"]
  last_mm[session]=v["mm_features"]
}
END {
  print "samples=" n
  print "sessions=" length(last_first_hash)
  print "first_hash_changed=" first_hash_changed+0
  print "cache_salt_changed=" cache_salt_changed+0
  print "lora_changed=" lora_changed+0
  print "mm_changed=" mm_changed+0
  print "skip_prefix_read=" skip_prefix_read+0
}'
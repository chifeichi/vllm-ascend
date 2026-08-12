grep "VERL_PD_SESSION_CACHE]" 1.log |
awk '
{
  delete v
  for (i=1; i<=NF; i++) {
    split($i, a, "=")
    v[a[1]]=a[2]
  }
  if (v["turn"] >= 2) {
    n++
    prompt += v["prompt_tokens"]
    cached += v["p_cached_tokens"]
    identical += v["identical_previous_tokens"]
    prev_decode += v["previous_decode_tokens"]
    identical_decode += v["identical_previous_decode_tokens"]
    missing += v["missing_identical_tokens"]
    missing_decode += v["missing_identical_decode_tokens"]
  }
}
END {
  print "samples=" n
  print "avg_prompt=" prompt/n
  print "avg_cached=" cached/n
  print "avg_identical_previous=" identical/n
  print "avg_previous_decode=" prev_decode/n
  print "avg_identical_previous_decode=" identical_decode/n
  print "avg_missing_identical=" missing/n
  print "avg_missing_identical_decode=" missing_decode/n
  print "actual_hit_rate=" 100*cached/prompt "%"
  print "identical_prefix_ratio=" 100*identical/prompt "%"
  print "missing_decode_share=" 100*missing_decode/missing "%"
}'
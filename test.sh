grep -a '\[ROLLOUT_SAMPLE\]' <运行日志> \
| sed 's/^.*\[ROLLOUT_SAMPLE\] //' \
| jq -s -r '
  map(select(
    .instance_id != "" and
    .claude_code_exit_code == 0 and
    .found_eval_status == true
  ))
  | group_by(.instance_id)
  | map({
      instance_id: .[0].instance_id,
      model_tokens: (map(.model_tokens) | add),
      prompt_tokens: (map(.prompt_tokens) | add)
    })
  | sort_by(-.model_tokens)
  | .[:64]
  | .[].instance_id
' > hard64_ids.txt


python3 -c 'import pandas as pd; ids=[x.strip() for x in open("hard64_ids.txt") if x.strip()]; rank={v:i for i,v in enumerate(ids)}; df=pd.read_parquet("<hard200.parquet>"); out=df[df["instance_id"].isin(rank)].copy(); out["_rank"]=out["instance_id"].map(rank); out.sort_values("_rank").drop(columns="_rank").to_parquet("swe_rebench_long_decode_64.parquet",index=False); print(len(out))'
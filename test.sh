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

python3 - <<'PY'
import pandas as pd
from filter_swe_rebench_hard import get_nested_metadata

input_file = "<hard200.parquet>"
output_file = "swe_rebench_long_decode_64.parquet"

ids = [line.strip() for line in open("hard64_ids.txt") if line.strip()]
rank = {instance_id: i for i, instance_id in enumerate(ids)}

df = pd.read_parquet(input_file)

if "instance_id" in df.columns:
    instance_ids = df["instance_id"].astype(str)
else:
    instance_ids = (
        df["extra_info"]
        .map(get_nested_metadata)
        .map(lambda metadata: str(metadata.get("instance_id", "")))
    )

selected = df[instance_ids.isin(rank)].copy()
selected["_rank"] = instance_ids[selected.index].map(rank)
selected = selected.sort_values("_rank").drop(columns="_rank")
selected.to_parquet(output_file, index=False)

print(f"selected={len(selected)} output={output_file}")

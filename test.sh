python - <<'PY'
import pandas as pd

x = pd.read_csv("swe_rebench_pd_selection.csv")

for c in [
    "rollout_n_eligible",
    "trajectory_eligible",
    "tail_eligible",
    "cache_eligible",
    "turns_eligible",
    "ratio_eligible",
    "selection_eligible",
]:
    ok = x[c].astype(str).str.lower().eq("true")
    print(c, ok.sum())
PY
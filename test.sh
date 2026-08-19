python - <<'PY'
import linecache
import megatron.core.transformer.moe.moe_utils as moe_utils

path = moe_utils.__file__
print("file:", path)

for line_no in range(325, 341):
    print(f"{line_no}: {linecache.getline(path, line_no).rstrip()}")
PY
MS_DIR="$(python - <<'PY'
import mindspeed
from pathlib import Path
print(Path(mindspeed.__file__).parent)
PY
)"

grep -RnsE --include='*.py' \
  'permute_with_probs|moe_token_permute|fused_permute|HAVE_TE' \
  "$MS_DIR" | head -n 50
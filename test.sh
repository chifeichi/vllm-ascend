MS_DIR="$(python - <<'PY'
import mindspeed
from pathlib import Path
print(Path(mindspeed.__file__).parent)
PY
)"

grep -Rns --include='*.py' \
  'fused_permute_with_probs' \
  "$MS_DIR" | head -n 30
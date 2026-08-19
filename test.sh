python - <<'PY'
import site
from pathlib import Path
for root in site.getsitepackages():
    for p in Path(root).glob("*megatron_core*"):
        print(p)
PY
python - <<'PY'
import importlib.metadata as md
for d in md.distributions():
    if (d.metadata.get("Name") or "").lower().replace("_", "-") == "megatron-core":
        print(d.version, d._path)
PY
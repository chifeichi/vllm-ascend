python - <<'PY'
import importlib.metadata as md
import megatron

for dist in md.distributions():
    if (dist.metadata.get("Name") or "").lower().replace("_", "-") == "megatron-core":
        print("distribution:", dist.version, dist._path)

print("megatron paths:", list(megatron.__path__))
PY
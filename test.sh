python - <<'PY'
import inspect
import megatron.core.transformer.moe.moe_utils as moe_utils

permute = moe_utils.permute
fused = permute.__globals__.get("fused_permute_with_probs")

print("permute module:", permute.__module__)
print("permute file:", inspect.getsourcefile(permute))
print("fused function:", fused)
print("fused module:", getattr(fused, "__module__", None))
print("fused file:", inspect.getsourcefile(fused) if fused else None)
print("fused signature:", inspect.signature(fused) if fused else None)
PY
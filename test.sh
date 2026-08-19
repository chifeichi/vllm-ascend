python - <<'PY'
import inspect
import importlib.metadata as md
import megatron.core.transformer.moe.moe_utils as moe_utils

for package in ("megatron-core", "mindspeed", "torch", "torch-npu"):
    try:
        print(package, md.version(package))
    except Exception as e:
        print(package, e)

func = moe_utils.fused_permute_with_probs
print("moe_utils:", moe_utils.__file__)
print("function module:", inspect.getmodule(func).__file__)
print("signature:", inspect.signature(func))
PY
(
OPS_DIR=$(python -c 'import vllm_ascend; from pathlib import Path; print(Path(vllm_ascend.__file__).resolve().parent / "_cann_ops_custom")')
mv "$OPS_DIR" "${OPS_DIR}.disabled"
trap 'mv "${OPS_DIR}.disabled" "$OPS_DIR"' EXIT

export ASCEND_CUSTOM_OPP_PATH="$(printf '%s' "$ASCEND_CUSTOM_OPP_PATH" | tr ':' '\n' | grep -v '_cann_ops_custom' | paste -sd: -)"
export LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '_cann_ops_custom' | paste -sd: -)"

python test_thd.py --case short --device 0 --tp 1 --cp 1
)
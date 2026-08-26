mkdir -p /opt/qwen35-thd
cd /opt/qwen35-thd

git clone --depth 1 -b core_v0.18.0 \
  https://github.com/NVIDIA/Megatron-LM.git

git clone --depth 1 -b core_r0.18.0 \
  https://gitcode.com/Ascend/MegatronAdaptor.git

git clone --depth 1 \
  https://gitcode.com/Ascend/TransformerEngineNPU.git

git clone --depth 1 -b core_r0.18.0 \
  https://gitcode.com/Ascend/MindSpeed.git

git clone --depth 1 -b v0.5.0 \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

git clone --depth 1 \
  https://gitcode.com/Ascend/MindSpeed-Ops.git

git clone --depth 1 \
  https://gitcode.com/Ascend/MindSpeed-Bridge.git

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

pip uninstall -y \
  megatron-core megatron-bridge \
  mindspeed megatron-adaptor \
  transformer-engine mindspeed-ops mindspeed-bridge

pip install decorator pybind11 diffusers

cd /opt/qwen35-thd/Megatron-LM
pip install -e . --no-deps

cd /opt/qwen35-thd/MegatronAdaptor
pip install -e . --no-deps

cd /opt/qwen35-thd/TransformerEngineNPU
pip install -e . --no-deps

cd /opt/qwen35-thd/MindSpeed
pip install -e . --no-deps

cd /opt/qwen35-thd/Megatron-Bridge
pip install -e . --no-deps

cd /opt/qwen35-thd/MindSpeed-Ops
pip install -e . \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple \
  --no-build-isolation --no-deps

cd /opt/qwen35-thd/MindSpeed-Bridge
pip install -e . --no-deps


#!/usr/bin/env python3
"""NPU smoke test for Qwen3.5-35B packed (THD) GDN kernels.

The short case checks both forward and backward.  The long case reproduces the
69,632-token shape seen in rollout training and checks forward separately so a
native operator crash can be attributed to that exact case.
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    backward: bool


CASES = {
    "short": Case("short", (1024, 2048), True),
    "long69632": Case("long69632", (32768, 36864), False),
}


def _run_case(case: Case, device_id: int) -> None:
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401 - registers the NPU backend

    try:
        import fla_npu  # noqa: F401 - registers FLA Ascend operators
    except ImportError as exc:
        raise RuntimeError(
            "fla_npu cannot be imported; flash-linear-attention-npu is not active"
        ) from exc

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    signature = inspect.signature(chunk_gated_delta_rule)
    if "cu_seqlens" not in signature.parameters:
        raise RuntimeError(
            "The active chunk_gated_delta_rule has no cu_seqlens argument; "
            "this FLA installation cannot exercise THD"
        )

    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    torch.manual_seed(1234)

    # Qwen3.5-35B-A3B uses 32 value heads. Its 16 key heads are repeated to 32
    # before GDN, and both the key and value head dimensions are 128.
    batch = 1
    total_tokens = sum(case.lengths)
    heads = 32
    key_dim = 128
    value_dim = 128
    dtype = torch.bfloat16
    requires_grad = case.backward

    def randn(*shape: int):
        return torch.randn(
            *shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    q = randn(batch, total_tokens, heads, key_dim)
    k = randn(batch, total_tokens, heads, key_dim)
    v = randn(batch, total_tokens, heads, value_dim)
    g = (-torch.rand(batch, total_tokens, heads, device=device)).requires_grad_(
        requires_grad
    )
    beta = torch.rand(
        batch, total_tokens, heads, device=device, dtype=dtype
    ).requires_grad_(requires_grad)

    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int64)

    print(
        f"CASE={case.name} phase=start lengths={list(case.lengths)} "
        f"total_tokens={total_tokens} q_shape={tuple(q.shape)} "
        f"cu_seqlens={offsets} backward={case.backward}",
        flush=True,
    )
    print(
        f"GDN_SOURCE={inspect.getfile(chunk_gated_delta_rule)}",
        flush=True,
    )
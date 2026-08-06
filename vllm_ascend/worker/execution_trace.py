# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os

import torch


def execution_trace(stage: str, *, synchronize: bool = False, **fields: object) -> None:
    """Print a flush-on-write marker and optionally synchronize the NPU."""
    details = " ".join(f"{name}={value!r}" for name, value in fields.items())
    prefix = (
        f"[VLLM_ASCEND_EXEC_TRACE pid={os.getpid()} "
        f"rank={os.getenv('RANK', '?')}]"
    )
    print(f"{prefix} {stage}{' ' + details if details else ''}", flush=True)

    if synchronize:
        print(f"{prefix} {stage}:sync_begin", flush=True)
        torch.npu.synchronize()
        print(f"{prefix} {stage}:sync_end", flush=True)

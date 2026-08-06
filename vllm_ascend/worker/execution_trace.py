# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os

import torch

import vllm_ascend.envs as envs_ascend


def execution_trace(stage: str, *, synchronize: bool = False, **fields: object) -> None:
    """Print an opt-in, flush-on-write marker for worker crash diagnosis."""
    if not envs_ascend.VLLM_ASCEND_EXECUTION_TRACE:
        return

    details = " ".join(f"{name}={value!r}" for name, value in fields.items())
    prefix = (
        f"[VLLM_ASCEND_EXEC_TRACE pid={os.getpid()} "
        f"rank={os.getenv('RANK', '?')}]"
    )
    print(f"{prefix} {stage}{' ' + details if details else ''}", flush=True)

    if synchronize and envs_ascend.VLLM_ASCEND_EXECUTION_TRACE_SYNC:
        print(f"{prefix} {stage}:sync_begin", flush=True)
        torch.npu.synchronize()
        print(f"{prefix} {stage}:sync_end", flush=True)

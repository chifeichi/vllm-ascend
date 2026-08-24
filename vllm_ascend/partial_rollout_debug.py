"""Opt-in synchronization fences for diagnosing partial-rollout NPU faults."""

import os
import time
from typing import Any

import torch
from vllm.logger import logger


_DEBUG_ENV = "PARTIAL_ROLLOUT_DEBUG_SYNC"


def partial_rollout_debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def partial_rollout_debug_sync(stage: str, **details: Any) -> None:
    """Synchronize the current NPU and identify the first failing boundary.

    This is intentionally expensive and must only be enabled for diagnosis.
    The ``before_sync`` line proves that control reached the boundary; if no
    matching ``sync_ok`` line follows, the asynchronous failure happened at or
    before this boundary.
    """
    if not partial_rollout_debug_enabled():
        return

    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    try:
        device = torch.npu.current_device()
    except Exception:
        device = "unknown"
    process_context = (
        f"time_ns={time.time_ns()} pid={os.getpid()} device={device} "
        f"rank={os.environ.get('RANK', 'unknown')} local_rank={os.environ.get('LOCAL_RANK', 'unknown')} "
        f"replica_rank={os.environ.get('VERL_REPLICA_RANK', 'unknown')}"
    )
    logger.warning(
        "[PR_DEBUG] stage=%s phase=before_sync %s %s",
        stage,
        process_context,
        detail_text,
    )
    try:
        torch.npu.synchronize()
    except Exception:
        logger.exception(
            "[PR_DEBUG] stage=%s phase=sync_failed %s %s",
            stage,
            process_context,
            detail_text,
        )
        raise

    memory_text = ""
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        memory_text = f" free_bytes={free_bytes} total_bytes={total_bytes}"
    except Exception as exc:
        memory_text = f" memory_info_error={type(exc).__name__}:{exc}"
    logger.warning(
        "[PR_DEBUG] stage=%s phase=sync_ok %s%s %s",
        stage,
        process_context,
        memory_text,
        detail_text,
    )

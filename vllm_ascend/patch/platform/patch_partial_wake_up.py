# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCore

logger = init_logger(__name__)


def _debug_wake(stage: str, **details: Any) -> None:
    if os.environ.get("PARTIAL_ROLLOUT_DEBUG_SYNC", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    logger.warning(
        "[PR_DEBUG] stage=%s time_ns=%s pid=%s replica_rank=%s %s",
        stage,
        time.time_ns(),
        os.getpid(),
        os.environ.get("VERL_REPLICA_RANK", "unknown"),
        detail_text,
    )


def _wake_up_without_early_scheduler_resume(
    self: EngineCore, tags: list[str] | None = None
) -> None:
    """Keep scheduling paused until all sleep-mode allocations are awake.

    vLLM supports waking weights and KV cache in separate stages. The generic
    EngineCore implementation resumes scheduling after every wake_up call,
    including a weights-only wake. CaMem has not remapped the KV cache at that
    point, so a queued partial-rollout request can execute against unmapped NPU
    memory.

    Preserve the explicit ``scheduling`` wake behavior, but otherwise resume
    only after the executor reports that no allocation tags remain asleep.
    """
    sleeping_tags_before = getattr(self.model_executor, "sleeping_tags", "unknown")
    _debug_wake(
        "engine_core_wake_entry",
        tags=tags or "all",
        executor_is_sleeping=self.model_executor.is_sleeping,
        sleeping_tags=sleeping_tags_before,
    )

    resume_scheduling = tags is not None and "scheduling" in tags
    if resume_scheduling:
        tags = [tag for tag in tags if tag != "scheduling"]

    if tags is None or tags:
        _debug_wake(
            "engine_core_wake_pre_executor",
            tags=tags or "all",
            sleeping_tags=getattr(self.model_executor, "sleeping_tags", "unknown"),
        )
        try:
            self.model_executor.wake_up(tags)
        except Exception:
            logger.exception(
                "[PR_DEBUG] stage=engine_core_wake_executor_failed "
                "time_ns=%s pid=%s replica_rank=%s tags=%s sleeping_tags=%s",
                time.time_ns(),
                os.getpid(),
                os.environ.get("VERL_REPLICA_RANK", "unknown"),
                tags or "all",
                getattr(self.model_executor, "sleeping_tags", "unknown"),
            )
            raise
        _debug_wake(
            "engine_core_wake_post_executor",
            tags=tags or "all",
            executor_is_sleeping=self.model_executor.is_sleeping,
            sleeping_tags=getattr(self.model_executor, "sleeping_tags", "unknown"),
        )

    if resume_scheduling or not self.model_executor.is_sleeping:
        _debug_wake(
            "engine_core_wake_resume_scheduler",
            tags=tags or "all",
            explicit_scheduling=resume_scheduling,
        )
        self.resume_scheduler()
    else:
        _debug_wake(
            "engine_core_wake_keep_scheduler_paused",
            tags=tags or "all",
            sleeping_tags=getattr(self.model_executor, "sleeping_tags", "unknown"),
        )


EngineCore.wake_up = _wake_up_without_early_scheduler_resume

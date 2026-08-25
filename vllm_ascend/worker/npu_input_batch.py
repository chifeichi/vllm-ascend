#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_input_batch.py
#

from typing import Any, cast

import numpy as np
import torch
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.pool.metadata import PoolingStates
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import copy_slice
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.partial_rollout_debug import partial_rollout_debug_sync
from vllm_ascend.worker.block_table import MultiGroupBlockTable


class NPUInputBatch(InputBatch):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[list[int]],
        max_num_blocks_per_req: list[int] | None = None,
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: list[KVCacheGroupSpec] | None = None,
    ):
        self.is_pooling_model = is_pooling_model
        self.is_spec_decode = is_spec_decode
        # Added for compatibility with InputBatch methods that reference these
        # attributes after PR vllm-project/vllm#34668. NPU does not use
        # thinking budget, so the holder is always None.
        self.thinking_budget_state_holder = None
        self.thinking_token_budget_reqs: set[str] = set()
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len), device="cpu", dtype=bool, pin_memory=False
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()
        # Store prompt embeddings per request to avoid OOM from large upfront
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            num_speculative_tokens=num_speculative_tokens,
            kernel_sizes=kernel_block_sizes,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            kv_cache_groups=kv_cache_groups,
        )

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Speculative decoding
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}

        # To accumulate prompt logprobs tensor chunks across prefill steps.
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # req_id -> list of specific token IDs to compute logprobs for
        # More efficient than num_logprobs=-1 when only a few tokens are needed
        self.logprob_token_ids: dict[str, list[int]] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

        self.req_output_token_ids: list[list[int] | None] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids

        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

    def add_request(self, request: Any) -> int:
        partial_rollout_debug_sync(
            "input_batch_add_request_pre",
            req_id=getattr(request, "req_id", "unknown"),
            num_reqs=self.num_reqs,
        )
        req_index = super().add_request(request)
        partial_rollout_debug_sync(
            "input_batch_add_request_post",
            req_id=getattr(request, "req_id", "unknown"),
            req_index=req_index,
            num_reqs=self.num_reqs,
        )
        return req_index

    def refresh_metadata(self) -> None:
        partial_rollout_debug_sync(
            "input_batch_refresh_metadata_pre",
            num_reqs=self.num_reqs,
        )

        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                partial_rollout_debug_sync(
                    "input_batch_sampling_metadata_pre_build",
                    num_reqs=self.num_reqs,
                    pooling=True,
                )
                self.sampling_metadata = self._make_sampling_metadata()
                partial_rollout_debug_sync(
                    "input_batch_sampling_metadata_post_build",
                    num_reqs=self.num_reqs,
                    pooling=True,
                )
            return

        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        if self.thinking_budget_state_holder is not None and batch_update:
            partial_rollout_debug_sync(
                "input_batch_thinking_budget_pre_sync",
                num_reqs=self.num_reqs,
            )
            self.thinking_budget_state_holder.sync_batch(batch_update)
            partial_rollout_debug_sync(
                "input_batch_thinking_budget_post_sync",
                num_reqs=self.num_reqs,
            )

        logit_procs = tuple(self.logitsprocs.all)
        partial_rollout_debug_sync(
            "input_batch_logitsprocs_pre_update",
            num_reqs=self.num_reqs,
            processor_count=len(logit_procs),
        )
        for logit_proc in logit_procs:
            logit_proc.update_state(batch_update)
        partial_rollout_debug_sync(
            "input_batch_logitsprocs_post_update",
            num_reqs=self.num_reqs,
            processor_count=len(logit_procs),
        )

        if batch_update:
            partial_rollout_debug_sync(
                "input_batch_sampling_metadata_pre_build",
                num_reqs=self.num_reqs,
                pooling=False,
            )
            self.sampling_metadata = self._make_sampling_metadata()
            partial_rollout_debug_sync(
                "input_batch_sampling_metadata_post_build",
                num_reqs=self.num_reqs,
                pooling=False,
            )
        partial_rollout_debug_sync(
            "input_batch_refresh_metadata_post",
            num_reqs=self.num_reqs,
        )

    def _make_sampling_metadata(self) -> SamplingMetadata:
        """Build sampling metadata with fine-grained NPU sync diagnostics."""
        num_reqs = self.num_reqs

        partial_rollout_debug_sync(
            "sampling_metadata_temperature_pre", num_reqs=num_reqs
        )
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
        partial_rollout_debug_sync(
            "sampling_metadata_temperature_post", num_reqs=num_reqs
        )

        if not self.no_top_p:
            partial_rollout_debug_sync(
                "sampling_metadata_top_p_pre", num_reqs=num_reqs
            )
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
            partial_rollout_debug_sync(
                "sampling_metadata_top_p_post", num_reqs=num_reqs
            )
        if not self.no_top_k:
            partial_rollout_debug_sync(
                "sampling_metadata_top_k_pre", num_reqs=num_reqs
            )
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
            partial_rollout_debug_sync(
                "sampling_metadata_top_k_post", num_reqs=num_reqs
            )

        if not self.no_penalties:
            partial_rollout_debug_sync(
                "sampling_metadata_penalties_pre", num_reqs=num_reqs
            )
            copy_slice(
                self.frequency_penalties_cpu_tensor,
                self.frequency_penalties,
                num_reqs,
            )
            copy_slice(
                self.presence_penalties_cpu_tensor,
                self.presence_penalties,
                num_reqs,
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )
            partial_rollout_debug_sync(
                "sampling_metadata_penalties_post", num_reqs=num_reqs
            )

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        prompt_token_ids_cpu = (
            self._make_prompt_token_ids_cpu_tensor()
            if needs_prompt_token_ids
            else None
        )
        partial_rollout_debug_sync(
            "sampling_metadata_prompt_tokens_pre_h2d",
            num_reqs=num_reqs,
            enabled=prompt_token_ids_cpu is not None,
        )
        prompt_token_ids = (
            prompt_token_ids_cpu.to(device=self.device, non_blocking=True)
            if prompt_token_ids_cpu is not None
            else None
        )
        partial_rollout_debug_sync(
            "sampling_metadata_prompt_tokens_post_h2d",
            num_reqs=num_reqs,
            enabled=prompt_token_ids_cpu is not None,
        )

        holder = self.thinking_budget_state_holder
        thinking_budget_tracks_reqs = (
            holder is not None and holder.has_tracked_requests()
        )
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
            or thinking_budget_tracks_reqs
        )
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )

        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            partial_rollout_debug_sync(
                "sampling_metadata_allowed_mask_pre", num_reqs=num_reqs
            )
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]
            partial_rollout_debug_sync(
                "sampling_metadata_allowed_mask_post", num_reqs=num_reqs
            )

        logprob_token_ids_by_index: dict[int, list[int]] | None = None
        if self.logprob_token_ids:
            logprob_token_ids_by_index = {}
            for req_id, token_ids in self.logprob_token_ids.items():
                if req_id in self.req_id_to_index:
                    req_index = self.req_id_to_index[req_id]
                    logprob_token_ids_by_index[req_index] = token_ids

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            logprob_token_ids=logprob_token_ids_by_index,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=output_token_ids,
            spec_token_ids=self.spec_token_ids,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
            thinking_budget_state_holder=self.thinking_budget_state_holder,
        )

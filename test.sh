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
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    backward: bool


CASES = {
    "short": Case("short", (1024, 2048), True),
    "long69632": Case("long69632", (32768, 36864), False),
}


def _git_revision(source_file: str) -> str:
    path = Path(source_file).resolve().parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    return "unknown"


def _run_case(case: Case, device_id: int, tp_size: int, cp_size: int) -> None:
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401 - registers the NPU backend

    try:
        import fla_npu  # noqa: F401 - registers FLA Ascend operators
    except ImportError as exc:
        raise RuntimeError(
            "fla_npu cannot be imported; flash-linear-attention-npu is not active"
        ) from exc

    # Activate the same MindSpeed patch path used by Megatron workers.  The
    # separate MegatronAdaptor package exists in the newer stack; core_r0.16.0
    # uses mindspeed.megatron_adaptor instead.
    try:
        import megatron_adaptor  # noqa: F401
    except ImportError:
        import mindspeed.megatron_adaptor  # noqa: F401

    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    chunk_gated_delta_rule = mindspeed_gdn.chunk_gated_delta_rule

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
    # before GDN. TP first shards the heads, then MindSpeed's CP->HP all-to-all
    # trades another CP factor of heads for the full packed sequence.
    batch = 1
    total_tokens = sum(case.lengths)
    if 32 % (tp_size * cp_size) != 0:
        raise ValueError(
            f"32 value heads are not divisible by TP*CP={tp_size * cp_size}"
        )
    heads = 32 // (tp_size * cp_size)
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
    # Match MindSpeed GatedDeltaNet._prepare_qkv_for_gated_delta_rule(): q/k
    # are normalized by FLA first, then all operator inputs are contiguous.
    q = mindspeed_gdn.l2norm(q.contiguous()).detach().requires_grad_(requires_grad)
    k = mindspeed_gdn.l2norm(k.contiguous()).detach().requires_grad_(requires_grad)
    v = v.contiguous().detach().requires_grad_(requires_grad)

    # Match GatedDeltaNet._compute_g_and_beta(), including g being promoted to
    # fp32 by alpha.float() while beta remains in the model dtype.
    alpha = torch.randn(batch, total_tokens, heads, device=device, dtype=dtype)
    a_log = torch.rand(heads, device=device, dtype=dtype) * 15 + 1
    a_log = a_log.log()
    dt_bias = torch.ones(heads, device=device, dtype=dtype)
    g = (-a_log.exp() * functional.softplus(alpha.float() + dt_bias)).contiguous()
    g = g.detach().requires_grad_(requires_grad)
    beta = torch.randn(batch, total_tokens, heads, device=device, dtype=dtype)
    beta = beta.sigmoid().contiguous().detach().requires_grad_(requires_grad)

    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    # VERL's preprocess_thd_engine creates cu_seqlens_padded as int32.
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)

    print(
        f"CASE={case.name} phase=start lengths={list(case.lengths)} "
        f"total_tokens={total_tokens} q_shape={tuple(q.shape)} "
        f"cu_seqlens={offsets} cu_dtype={cu_seqlens.dtype} "
        f"tp={tp_size} cp={cp_size} local_heads={heads} backward={case.backward}",
        flush=True,
    )
    print(
        f"ACTIVE_GDN_CLASS={inspect.getfile(GatedDeltaNet)} "
        f"GDN_CALLABLE={inspect.getfile(chunk_gated_delta_rule)} "
        f"HAVE_FLA={mindspeed_gdn.HAVE_FLA} "
        f"MINDSPEED_REV={_git_revision(inspect.getfile(mindspeed_gdn))}",
        flush=True,
    )

    print(f"CASE={case.name} phase=pre_gdn_call", flush=True)
    output, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    print(f"CASE={case.name} phase=post_gdn_call_pre_sync", flush=True)
    torch.npu.synchronize()
    print(f"CASE={case.name} phase=post_gdn_sync", flush=True)

    expected_shape = (batch, total_tokens, heads, value_dim)
    if tuple(output.shape) != expected_shape:
        raise AssertionError(
            f"unexpected output shape: {tuple(output.shape)} != {expected_shape}"
        )
    if final_state is not None:
        raise AssertionError("final_state must be None when output_final_state=False")
    if not bool(torch.isfinite(output).all().item()):
        raise AssertionError("GDN forward produced NaN or Inf")

    print(f"CASE={case.name} phase=forward PASS", flush=True)

    if case.backward:
        loss = functional.mse_loss(output.float(), torch.zeros_like(output.float()))
        loss.backward()
        torch.npu.synchronize()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.grad is None:
                raise AssertionError(f"{name}.grad is None")
            if not bool(torch.isfinite(tensor.grad).all().item()):
                raise AssertionError(f"{name}.grad contains NaN or Inf")
        print(f"CASE={case.name} phase=backward PASS", flush=True)


def _run_suite(device_id: int, skip_long: bool, tp_size: int, cp_size: int) -> int:
    selected = ["short"] if skip_long else ["short", "long69632"]
    failed = []
    script = os.path.abspath(__file__)

    for name in selected:
        command = [
            sys.executable,
            script,
            "--case",
            name,
            "--device",
            str(device_id),
            "--tp",
            str(tp_size),
            "--cp",
            str(cp_size),
        ]
        print(f"\n===== RUN {name} =====", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed.append((name, result.returncode))

    if failed:
        print(f"\nTHD_GDN_RESULT=FAIL failed_cases={failed}", flush=True)
        return 1

    print("\nTHD_GDN_RESULT=PASS", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="Only run the short forward/backward THD case",
    )
    args = parser.parse_args()

    if args.case:
        _run_case(CASES[args.case], args.device, args.tp, args.cp)
        print(f"CASE={args.case} RESULT=PASS", flush=True)
        return 0
    return _run_suite(args.device, args.skip_long, args.tp, args.cp)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Filter processed SWE-rebench and add post_setup_cmd to two SWE datasets."""

import argparse
import copy
import json
import re
from pathlib import Path

import pandas as pd


def as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"Expected a dict, got {type(value).__name__}")


def get_tools_kwargs(extra_info: object) -> dict:
    return as_dict(as_dict(extra_info).get("tools_kwargs"))


def get_metadata(extra_info: object) -> dict:
    tools_kwargs = get_tools_kwargs(extra_info)

    reward = tools_kwargs.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("metadata"), dict):
        return reward["metadata"]

    task = tools_kwargs.get("task")
    if isinstance(task, dict) and isinstance(task.get("metadata"), dict):
        return task["metadata"]

    raise KeyError(
        "Cannot find metadata under extra_info.tools_kwargs.reward.metadata "
        "or extra_info.tools_kwargs.task.metadata"
    )


def count_patch_lines(patch: object) -> int:
    if not isinstance(patch, str):
        return 0
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )


def count_patch_files(patch: object) -> int:
    if not isinstance(patch, str):
        return 0
    return len(re.findall(r"^diff --git ", patch, flags=re.MULTILINE))


def count_items(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return int(bool(value))
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return int(bool(value))


def select_hard_rebench(dataset: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    if num_samples <= 0 or num_samples > len(dataset):
        raise ValueError(
            f"--num-rebench must be within [1, {len(dataset)}], got {num_samples}"
        )

    metadata = dataset["extra_info"].map(get_metadata)
    patch = metadata.map(lambda item: item.get("patch", ""))
    fail_to_pass = metadata.map(lambda item: item.get("FAIL_TO_PASS", []))
    problem = metadata.map(lambda item: item.get("problem_statement", ""))

    metrics = pd.DataFrame(index=dataset.index)
    metrics["patch_lines"] = patch.map(count_patch_lines)
    metrics["patch_files"] = patch.map(count_patch_files)
    metrics["fail_tests"] = fail_to_pass.map(count_items)
    metrics["problem_chars"] = problem.map(
        lambda value: len(value) if isinstance(value, str) else 0
    )

    # Proxy for difficult, long-horizon tasks. Percentile ranks prevent one
    # metric with extreme values from dominating the score.
    metrics["score"] = (
        0.45 * metrics["patch_lines"].rank(pct=True)
        + 0.25 * metrics["patch_files"].rank(pct=True)
        + 0.15 * metrics["fail_tests"].rank(pct=True)
        + 0.15 * metrics["problem_chars"].rank(pct=True)
    )

    indices = metrics.nlargest(num_samples, "score").index
    return dataset.loc[indices].copy()


def add_post_setup_cmd(extra_info: object, dataset_name: str) -> dict:
    result = copy.deepcopy(as_dict(extra_info))
    tools_kwargs = as_dict(result.get("tools_kwargs"))
    metadata = get_metadata(result)
    base_commit = metadata["base_commit"]

    env_value = tools_kwargs.get("env", {})
    env = as_dict(env_value) if env_value else {}

    # Preserve the image from the newer task/sandbox schema if env is absent.
    if "deployment" not in env:
        task = tools_kwargs.get("task")
        if isinstance(task, dict):
            sandbox = task.get("sandbox")
            if isinstance(sandbox, dict) and sandbox.get("image"):
                env["deployment"] = {"image": sandbox["image"]}

    if dataset_name == "swe_rebench":
        reset_cmds = [
            "git tag -d $(git tag -l) || true",
            "git reflog expire --expire=now --all || true",
            "git gc --prune=now || true",
            f"git checkout {base_commit} || true",
            "git clean -fdq || true",
        ]
    elif dataset_name == "swe_bench_verified":
        reset_cmds = [
            "cd /testbed",
            "git restore .",
            "git reset --hard",
            f"git checkout {base_commit}",
            "git clean -fdq",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    env["post_setup_cmd"] = " && ".join(reset_cmds)
    tools_kwargs["env"] = env
    result["tools_kwargs"] = tools_kwargs
    return result


def process_rebench(input_path: Path, output_path: Path, num_samples: int) -> None:
    dataset = pd.read_parquet(input_path)
    selected = select_hard_rebench(dataset, num_samples)
    selected["extra_info"] = selected["extra_info"].map(
        lambda value: add_post_setup_cmd(value, "swe_rebench")
    )
    selected.to_parquet(output_path, index=False)
    print(f"SWE-rebench: {len(dataset)} -> {len(selected)} rows: {output_path}")


def process_verified(input_path: Path, output_path: Path) -> None:
    dataset = pd.read_parquet(input_path)
    dataset["extra_info"] = dataset["extra_info"].map(
        lambda value: add_post_setup_cmd(value, "swe_bench_verified")
    )
    dataset.to_parquet(output_path, index=False)
    print(f"SWE-bench Verified: {len(dataset)} rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebench-input", type=Path, required=True)
    parser.add_argument("--verified-input", type=Path, required=True)
    parser.add_argument(
        "--rebench-output",
        type=Path,
        default=Path("swe_rebench_filtered_hard_200.parquet"),
    )
    parser.add_argument(
        "--verified-output",
        type=Path,
        default=Path("swe_bench_verified_post_setup.parquet"),
    )
    parser.add_argument("--num-rebench", type=int, default=200)
    args = parser.parse_args()

    process_rebench(args.rebench_input, args.rebench_output, args.num_rebench)
    process_verified(args.verified_input, args.verified_output)


if __name__ == "__main__":
    main()

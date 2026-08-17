#!/usr/bin/env python3
"""Add post_setup_cmd to processed SWE-rebench and SWE-bench Verified."""

import argparse
import copy
import json
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


def process_rebench(input_path: Path, output_path: Path) -> None:
    dataset = pd.read_parquet(input_path)
    dataset["extra_info"] = dataset["extra_info"].map(
        lambda value: add_post_setup_cmd(value, "swe_rebench")
    )
    dataset.to_parquet(output_path, index=False)
    print(f"SWE-rebench: {len(dataset)} rows: {output_path}")


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
        default=Path("swe_rebench_post_setup.parquet"),
    )
    parser.add_argument(
        "--verified-output",
        type=Path,
        default=Path("swe_bench_verified_post_setup.parquet"),
    )
    args = parser.parse_args()

    process_rebench(args.rebench_input, args.rebench_output)
    process_verified(args.verified_input, args.verified_output)


if __name__ == "__main__":
    main()

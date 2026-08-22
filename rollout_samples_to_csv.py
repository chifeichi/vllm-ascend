#!/usr/bin/env python3
"""Convert ``[ROLLOUT_SAMPLE]`` log records into an n-aware CSV report.

The report has one row per ``instance_id``.  ``rollout.n`` is treated as
repeated observations used to assess stability, not as a token multiplier.

Example:

    python3 rollout_samples_to_csv.py \
        --log /path/to/train.log \
        --rollout-n 4 \
        --output rollout_samples_n4.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


MARKER = "[ROLLOUT_SAMPLE]"
ERROR_REASON = re.compile(
    r"error|timeout|failed|cancel|truncat|max.?length|length.?limit",
    re.IGNORECASE,
)

BASE_METRICS = (
    "num_turns",
    "prompt_tokens",
    "response_tokens",
    "model_tokens",
    "total_tokens",
)

DERIVED_METRICS = (
    "non_model_tokens",
    "final_context_tokens",
    "model_tokens_per_turn",
    "response_prompt_ratio",
    "model_prompt_ratio",
    "model_response_share",
    "repeated_model_tokens_proxy",
    "p_work_proxy",
    "total_work_proxy",
    "decode_pressure_proxy",
)

SUMMARY_METRICS = BASE_METRICS + DERIVED_METRICS


def parse_logs(paths: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    decoder = json.JSONDecoder()

    for path_string in paths:
        path = Path(path_string)
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                payload = line.split(MARKER, 1)[1].lstrip()
                try:
                    record, _ = decoder.raw_decode(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid {MARKER} JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object after {MARKER} at {path}:{line_number}"
                    )
                record["_log_path"] = str(path)
                record["_log_line"] = line_number
                records.append(record)

    if not records:
        raise ValueError(f"No {MARKER} records found in: {paths}")

    frame = pd.DataFrame(records)
    required = {
        "instance_id",
        "session_id",
        "trajectory_index",
        "num_turns",
        "prompt_tokens",
        "response_tokens",
        "model_tokens",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    frame["instance_id"] = frame["instance_id"].fillna("").astype(str)
    frame = frame[frame["instance_id"] != ""].copy()
    if frame.empty:
        raise ValueError("All ROLLOUT_SAMPLE records have an empty instance_id")

    # The same Ray log line can be collected more than once when users merge
    # driver and worker logs.  Keep the last copy of each trajectory.
    frame = frame.drop_duplicates(
        ["session_id", "trajectory_index"], keep="last"
    ).copy()

    if "total_tokens" not in frame:
        frame["total_tokens"] = (
            pd.to_numeric(frame["prompt_tokens"], errors="coerce").fillna(0)
            + pd.to_numeric(frame["response_tokens"], errors="coerce").fillna(0)
        )

    for name in BASE_METRICS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)

    frame["num_turns"] = frame["num_turns"].clip(lower=1.0)
    frame["non_model_tokens"] = (
        frame["response_tokens"] - frame["model_tokens"]
    ).clip(lower=0.0)
    frame["final_context_tokens"] = (
        frame["prompt_tokens"] + frame["response_tokens"]
    )
    frame["model_tokens_per_turn"] = (
        frame["model_tokens"] / frame["num_turns"].clip(lower=1.0)
    )
    frame["response_prompt_ratio"] = (
        frame["response_tokens"] / frame["prompt_tokens"].clip(lower=1.0)
    )
    frame["model_prompt_ratio"] = (
        frame["model_tokens"] / frame["prompt_tokens"].clip(lower=1.0)
    )
    frame["model_response_share"] = (
        frame["model_tokens"] / frame["response_tokens"].clip(lower=1.0)
    )

    # In multi-turn PD without D-to-P KV return, most model output from all but
    # the final turn becomes new P-side work later.  We only have trajectory
    # totals, so use the same even-per-turn approximation as the existing PD
    # sample selector.
    frame["repeated_model_tokens_proxy"] = (
        frame["model_tokens"]
        * (frame["num_turns"] - 1.0)
        / frame["num_turns"]
    )
    frame["p_work_proxy"] = (
        frame["prompt_tokens"]
        + frame["non_model_tokens"]
        + frame["repeated_model_tokens_proxy"]
    )
    frame["total_work_proxy"] = frame["p_work_proxy"] + frame["model_tokens"]
    frame["decode_pressure_proxy"] = (
        frame["model_tokens"] / frame["p_work_proxy"].clip(lower=1.0)
    )
    return frame


def normalized_bool_rate(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return math.nan
    values = group[column].dropna()
    if values.empty:
        return math.nan

    def normalize(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    return float(values.map(normalize).mean())


def exit_success_rate(group: pd.DataFrame) -> float:
    if "claude_code_exit_code" not in group:
        return math.nan
    values = pd.to_numeric(group["claude_code_exit_code"], errors="coerce").dropna()
    return float((values == 0).mean()) if not values.empty else math.nan


def normal_reason_rate(group: pd.DataFrame) -> float:
    if "materialization_reason" not in group:
        return math.nan
    values = group["materialization_reason"].dropna().astype(str)
    if values.empty:
        return math.nan
    return float((~values.str.contains(ERROR_REASON)).mean())


def json_values(series: pd.Series) -> str:
    values = [round(float(value), 6) for value in series]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def add_distribution(row: dict[str, object], group: pd.DataFrame, name: str) -> None:
    values = group[name].astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    row[f"{name}_values"] = json_values(values)
    row[f"{name}_min"] = float(values.min())
    row[f"{name}_p25"] = float(values.quantile(0.25))
    row[f"{name}_p50"] = float(values.median())
    row[f"{name}_mean"] = mean
    row[f"{name}_p75"] = float(values.quantile(0.75))
    row[f"{name}_max"] = float(values.max())
    row[f"{name}_std"] = std
    row[f"{name}_cv"] = std / mean if mean > 0 else math.nan


def summarize(frame: pd.DataFrame, rollout_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for instance_id, group in frame.groupby("instance_id", sort=False):
        group = group.sort_values(
            ["session_id", "trajectory_index"], kind="stable"
        )
        rollout_count = len(group)
        unique_sessions = int(group["session_id"].nunique())
        trajectories_per_session = group.groupby("session_id").size()
        complete_groups, remainder = divmod(rollout_count, rollout_n)

        row: dict[str, object] = {
            "instance_id": str(instance_id),
            "rollout_n_expected": rollout_n,
            "rollout_count": rollout_count,
            "unique_sessions": unique_sessions,
            "complete_n_groups": complete_groups,
            "incomplete_rollout_count": remainder,
            "has_complete_n": rollout_count >= rollout_n and remainder == 0,
            "exactly_one_complete_n": (
                rollout_count == rollout_n
                and unique_sessions == rollout_n
                and int(trajectories_per_session.max()) == 1
            ),
            "max_trajectories_per_session": int(trajectories_per_session.max()),
            "session_ids": json.dumps(
                group["session_id"].astype(str).tolist(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "exit_success_rate": exit_success_rate(group),
            "normal_reason_rate": normal_reason_rate(group),
            "eval_completed_rate": normalized_bool_rate(group, "eval_completed"),
            "found_eval_status_rate": normalized_bool_rate(
                group, "found_eval_status"
            ),
            "resolved_rate": normalized_bool_rate(group, "resolved"),
        }
        for name in SUMMARY_METRICS:
            add_distribution(row, group, name)

        # Conservative n-repeat signals used later for selection: a candidate
        # should remain decode-heavy in its lower quartile while its P/cache
        # work should remain bounded in its upper quartile.
        row["stable_decode_floor"] = row["model_tokens_per_turn_p25"]
        row["stable_model_share_floor"] = row["model_response_share_p25"]
        row["cache_pressure_ceiling"] = row["final_context_tokens_p75"]
        row["p_work_ceiling"] = row["p_work_proxy_p75"]
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["exactly_one_complete_n", "instance_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ROLLOUT_SAMPLE logs to one n-aware CSV row per instance"
    )
    parser.add_argument(
        "--log",
        nargs="+",
        required=True,
        help="One or more training log files containing [ROLLOUT_SAMPLE] records",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=4,
        help="Expected repeated rollouts per instance (default: 4)",
    )
    parser.add_argument(
        "--output",
        default="rollout_samples_n4.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    if args.rollout_n <= 0:
        parser.error("--rollout-n must be greater than zero")

    frame = parse_logs(args.log)
    report = summarize(frame, args.rollout_n)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    exact = int(report["exactly_one_complete_n"].sum())
    complete = int(report["has_complete_n"].sum())
    print(f"Parsed trajectory records: {len(frame)}")
    print(f"Unique instance_id values: {len(report)}")
    print(f"Exactly one complete n={args.rollout_n} group: {exact}")
    print(f"One or more complete n={args.rollout_n} groups: {complete}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()

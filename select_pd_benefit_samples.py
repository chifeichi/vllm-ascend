#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MARKER = "[ROLLOUT_SAMPLE]"
ERROR_REASON = re.compile(r"error|timeout|failed|cancel|truncat|max.?length|length.?limit", re.IGNORECASE)


def as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def nested_get(value: object, *path: str) -> object | None:
    current = value
    for key in path:
        current = as_dict(current).get(key)
    return current


def dataset_instance_id(row: pd.Series) -> str:
    candidates = (
        row.get("instance_id"),
        nested_get(row.get("extra_info"), "tools_kwargs", "reward", "metadata", "instance_id"),
        nested_get(row.get("extra_info"), "tools_kwargs", "task", "metadata", "instance_id"),
        nested_get(row.get("extra_info"), "instance_id"),
        nested_get(row.get("reward_model"), "ground_truth", "instance_id"),
    )
    for candidate in candidates:
        if candidate is not None and str(candidate):
            return str(candidate)
    return ""


def parse_logs(paths: list[str]) -> pd.DataFrame:
    records = []
    decoder = json.JSONDecoder()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                payload = line.split(MARKER, 1)[1].lstrip()
                try:
                    record, _ = decoder.raw_decode(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ROLLOUT_SAMPLE JSON at {path}:{line_number}: {exc}") from exc
                record["_log_path"] = path
                record["_log_line"] = line_number
                records.append(record)
    if not records:
        raise ValueError(f"No {MARKER} records found")
    frame = pd.DataFrame(records)
    required = {"instance_id", "num_turns", "prompt_tokens", "response_tokens"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROLLOUT_SAMPLE records are missing fields: {sorted(missing)}")
    dedup = [name for name in ("global_steps", "session_id", "trajectory_index") if name in frame.columns]
    if dedup:
        frame = frame.drop_duplicates(dedup, keep="last")
    frame = frame[frame["instance_id"].fillna("").astype(str) != ""].copy()
    for name in ("num_turns", "prompt_tokens", "response_tokens"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0)
    frame["num_turns"] = frame["num_turns"].clip(lower=1)
    frame["response_prompt_ratio"] = frame["response_tokens"] / frame["prompt_tokens"].clip(lower=1)
    return frame


def bool_rate(group: pd.DataFrame, name: str) -> float:
    if name not in group:
        return 1.0
    values = group[name].dropna()
    if values.empty:
        return 1.0
    normalized = values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes"}
    )
    return float(normalized.mean())


def exit_success_rate(group: pd.DataFrame) -> float:
    if "claude_code_exit_code" not in group:
        return 1.0
    values = pd.to_numeric(group["claude_code_exit_code"], errors="coerce").dropna()
    return float((values == 0).mean()) if not values.empty else 1.0


def normal_reason_rate(group: pd.DataFrame) -> float:
    if "materialization_reason" not in group:
        return 1.0
    values = group["materialization_reason"].dropna().astype(str)
    return float((~values.str.contains(ERROR_REASON)).mean()) if not values.empty else 1.0


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance_id, group in records.groupby("instance_id", sort=False):
        ratio = group["response_prompt_ratio"]
        prompt_mean = float(group["prompt_tokens"].mean())
        response_mean = float(group["response_tokens"].mean())
        turns_mean = float(group["num_turns"].mean())
        sessions = int(group["session_id"].nunique()) if "session_id" in group else len(group)
        rows.append(
            {
                "instance_id": str(instance_id),
                "sessions": sessions,
                "trajectory_records": len(group),
                "prompt_tokens_p25": float(group["prompt_tokens"].quantile(0.25)),
                "prompt_tokens_p50": float(group["prompt_tokens"].median()),
                "prompt_tokens_mean": prompt_mean,
                "response_tokens_min": float(group["response_tokens"].min()),
                "response_tokens_p25": float(group["response_tokens"].quantile(0.25)),
                "response_tokens_p50": float(group["response_tokens"].median()),
                "response_tokens_mean": response_mean,
                "response_tokens_max": float(group["response_tokens"].max()),
                "response_prompt_ratio_of_means": response_mean / max(prompt_mean, 1.0),
                "response_prompt_ratio_min": float(ratio.min()),
                "response_prompt_ratio_p25": float(ratio.quantile(0.25)),
                "response_prompt_ratio_p50": float(ratio.median()),
                "response_prompt_ratio_mean": float(ratio.mean()),
                "response_prompt_ratio_max": float(ratio.max()),
                "response_prompt_ratio_std": float(ratio.std(ddof=0)),
                "num_turns_mean": turns_mean,
                "num_turns_p50": float(group["num_turns"].median()),
                "num_turns_p90": float(group["num_turns"].quantile(0.90)),
                "response_tokens_per_turn": response_mean / max(turns_mean, 1.0),
                "found_eval_status_rate": bool_rate(group, "found_eval_status"),
                "eval_completed_rate": bool_rate(group, "eval_completed"),
                "resolved_rate": bool_rate(group, "resolved"),
                "exit_success_rate": exit_success_rate(group),
                "normal_reason_rate": normal_reason_rate(group),
            }
        )
    return pd.DataFrame(rows)


def rank_samples(
    summary: pd.DataFrame,
    min_ratio: float,
    max_turns_p50: float,
) -> pd.DataFrame:
    result = summary.copy()
    result["eligible"] = (
        (result["response_prompt_ratio_of_means"] >= min_ratio)
        & (result["num_turns_p50"] <= max_turns_p50)
    )
    eligible = result[result["eligible"]].sort_values(
        [
            "prompt_tokens_mean",
            "response_tokens_per_turn",
            "response_prompt_ratio_of_means",
            "response_tokens_mean",
        ],
        ascending=[True, False, False, False],
    )
    ineligible = result[~result["eligible"]].sort_values(
        ["response_prompt_ratio_of_means", "num_turns_p50", "prompt_tokens_mean"],
        ascending=[False, True, True],
    )
    ranked = pd.concat([eligible, ineligible], ignore_index=True)
    ranked["rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked.loc[ranked["eligible"], "rank"] = range(1, len(eligible) + 1)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select agent tasks with a large and stable response-to-prompt ratio"
    )
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="swe_rebench_pd64.parquet")
    parser.add_argument("--report", default="swe_rebench_pd_selection.csv")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--min-response-prompt-ratio", type=float, default=5.0)
    parser.add_argument("--max-turns-p50", type=float, default=20.0)
    args = parser.parse_args()

    dataset = pd.read_parquet(args.input)
    ids = dataset.apply(dataset_instance_id, axis=1)
    empty_id_count = int((ids == "").sum())
    if empty_id_count:
        raise ValueError(
            f"Cannot resolve instance_id for {empty_id_count} rows in the input parquet"
        )
    duplicate_ids = ids[ids != ""].value_counts()
    duplicate_ids = duplicate_ids[duplicate_ids > 1]
    if not duplicate_ids.empty:
        raise ValueError(f"Input parquet has duplicate instance_id values: {duplicate_ids.index[:10].tolist()}")

    all_records = parse_logs(args.log)
    input_ids = set(ids)
    records = all_records[all_records["instance_id"].isin(input_ids)].copy()
    ignored_records = len(all_records) - len(records)
    ignored_instances = all_records.loc[
        ~all_records["instance_id"].isin(input_ids), "instance_id"
    ].nunique()
    if records.empty:
        raise ValueError(
            "No ROLLOUT_SAMPLE records match instance_id values in the input parquet"
        )

    summary = rank_samples(
        summarize(records),
        args.min_response_prompt_ratio,
        args.max_turns_p50,
    )
    selected_summary = summary[summary["eligible"]].head(args.num_samples).copy()
    if len(selected_summary) < args.num_samples:
        raise ValueError(
            f"Only {len(selected_summary)} instances have response/prompt ratio >= "
            f"{args.min_response_prompt_ratio:g} and turns_p50 <= {args.max_turns_p50:g}, "
            f"fewer than --num-samples={args.num_samples}. Lower "
            "--min-response-prompt-ratio or raise --max-turns-p50."
        )
    summary["selected"] = summary["rank"].isin(selected_summary["rank"])

    rank = dict(zip(selected_summary["instance_id"], selected_summary["rank"], strict=True))
    selected = dataset[ids.isin(rank)].copy()
    selected["_selection_rank"] = ids[ids.isin(rank)].map(rank).to_numpy()
    selected = selected.sort_values("_selection_rank").drop(columns="_selection_rank")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    summary.to_csv(args.report, index=False)

    missing_logged_ids = sorted(input_ids - set(records["instance_id"]))
    prompt_mean = float(records["prompt_tokens"].mean())
    response_mean = float(records["response_tokens"].mean())
    print(f"ROLLOUT_SAMPLE records: {len(all_records)}")
    print(f"Matched records: {len(records)}")
    print(f"Ignored records outside input parquet: {ignored_records}")
    print(f"Ignored instances outside input parquet: {ignored_instances}")
    print(f"Logged sessions: {records['session_id'].nunique() if 'session_id' in records else len(records)}")
    print(f"Candidate instances: {len(summary)}")
    print(
        f"Instances with response/prompt ratio >= {args.min_response_prompt_ratio:g} "
        f"and turns_p50 <= {args.max_turns_p50:g}: "
        f"{int(summary['eligible'].sum())}"
    )
    print(f"Input instances without logs: {len(missing_logged_ids)}")
    if missing_logged_ids:
        print(f"Missing instance_id values: {missing_logged_ids}")
    print(f"All-record prompt mean: {prompt_mean:.3f}")
    print(f"All-record response mean: {response_mean:.3f}")
    print(f"All-record response/prompt ratio: {response_mean / max(prompt_mean, 1.0):.6f}")
    print(
        "Highest instance response/prompt ratio: "
        f"{summary['response_prompt_ratio_of_means'].max():.6f}"
    )
    print(f"Selected instances: {len(selected_summary)}")
    print(f"Output parquet: {args.output}")
    print(f"Selection report: {args.report}")
    for row in selected_summary.itertuples(index=False):
        print(
            f"rank={row.rank} instance_id={row.instance_id} "
            f"sessions={row.sessions} trajectories={row.trajectory_records} "
            f"turns_mean={row.num_turns_mean:.1f} turns_p50={row.num_turns_p50:.1f} "
            f"prompt_mean={row.prompt_tokens_mean:.0f} response_mean={row.response_tokens_mean:.0f} "
            f"response_per_turn={row.response_tokens_per_turn:.0f} "
            f"response_prompt_ratio={row.response_prompt_ratio_of_means:.3f} "
            f"exit_success_rate={row.exit_success_rate:.2f} resolved_rate={row.resolved_rate:.2f}"
        )


if __name__ == "__main__":
    main()


# python select_pd_benefit_samples.py \
#   --log 1.log \
#   --input swe_rebench_hard200.parquet \
#   --output swe_rebench_pd64.parquet \
#   --report swe_rebench_pd_selection.csv \
#   --num-samples 32 \
#   --min-response-prompt-ratio 5 \
#   --max-turns-p50 20

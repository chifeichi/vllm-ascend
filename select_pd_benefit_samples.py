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
    required = {
        "instance_id",
        "num_turns",
        "prompt_tokens",
        "response_tokens",
        "model_tokens",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROLLOUT_SAMPLE records are missing fields: {sorted(missing)}")
    dedup = [name for name in ("global_steps", "session_id", "trajectory_index") if name in frame.columns]
    if dedup:
        frame = frame.drop_duplicates(dedup, keep="last")
    frame = frame[frame["instance_id"].fillna("").astype(str) != ""].copy()
    for name in ("num_turns", "prompt_tokens", "response_tokens", "model_tokens"):
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
        session_key = "session_id" if "session_id" in group else None
        session_groups = group.groupby(session_key, sort=False) if session_key else [(None, group)]
        session_rows = []
        for _, session_group in session_groups:
            prompt_tokens = float(session_group["prompt_tokens"].sum())
            response_tokens = float(session_group["response_tokens"].sum())
            model_tokens = float(session_group["model_tokens"].sum())
            non_model_tokens = max(response_tokens - model_tokens, 0.0)
            num_turns = float(session_group["num_turns"].sum())

            # In multi-turn PD, model tokens from all but the final turn become
            # input to a later P request. With no D-to-P KV return path, use an
            # even-per-turn approximation for this repeated P-side work.
            trajectory_turns = session_group["num_turns"].clip(lower=1)
            repeated_model_tokens = float(
                (
                    session_group["model_tokens"]
                    * (trajectory_turns - 1.0)
                    / trajectory_turns
                ).sum()
            )
            p_work_proxy = prompt_tokens + non_model_tokens + repeated_model_tokens
            session_rows.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "model_tokens": model_tokens,
                    "non_model_tokens": non_model_tokens,
                    "num_turns": num_turns,
                    "p_work_proxy": p_work_proxy,
                    "decode_pressure_proxy": model_tokens / max(p_work_proxy, 1.0),
                    "model_tokens_per_turn": model_tokens / max(num_turns, 1.0),
                }
            )

        sessions_frame = pd.DataFrame(session_rows)
        prompt_mean = float(sessions_frame["prompt_tokens"].mean())
        response_mean = float(sessions_frame["response_tokens"].mean())
        model_mean = float(sessions_frame["model_tokens"].mean())
        non_model_mean = float(sessions_frame["non_model_tokens"].mean())
        turns_mean = float(sessions_frame["num_turns"].mean())
        p_work_mean = float(sessions_frame["p_work_proxy"].mean())
        sessions = len(sessions_frame)
        rows.append(
            {
                "instance_id": str(instance_id),
                "sessions": sessions,
                "trajectory_records": len(group),
                "prompt_tokens_p25": float(sessions_frame["prompt_tokens"].quantile(0.25)),
                "prompt_tokens_p50": float(sessions_frame["prompt_tokens"].median()),
                "prompt_tokens_mean": prompt_mean,
                "response_tokens_min": float(sessions_frame["response_tokens"].min()),
                "response_tokens_p25": float(sessions_frame["response_tokens"].quantile(0.25)),
                "response_tokens_p50": float(sessions_frame["response_tokens"].median()),
                "response_tokens_mean": response_mean,
                "response_tokens_max": float(sessions_frame["response_tokens"].max()),
                "model_tokens_mean": model_mean,
                "non_model_tokens_mean": non_model_mean,
                "response_prompt_ratio_of_means": response_mean / max(prompt_mean, 1.0),
                "model_prompt_ratio_of_means": model_mean / max(prompt_mean, 1.0),
                "num_turns_mean": turns_mean,
                "num_turns_p50": float(sessions_frame["num_turns"].median()),
                "num_turns_p90": float(sessions_frame["num_turns"].quantile(0.90)),
                "model_tokens_per_turn": float(sessions_frame["model_tokens_per_turn"].mean()),
                "p_work_proxy_mean": p_work_mean,
                "decode_pressure_proxy": model_mean / max(p_work_mean, 1.0),
                "found_eval_status_rate": bool_rate(group, "found_eval_status"),
                "eval_completed_rate": bool_rate(group, "eval_completed"),
                "resolved_rate": bool_rate(group, "resolved"),
                "exit_success_rate": exit_success_rate(group),
                "normal_reason_rate": normal_reason_rate(group),
            }
        )
    return pd.DataFrame(rows)


def rank_samples(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["decode_pressure_percentile"] = result["decode_pressure_proxy"].rank(
        method="average", pct=True, ascending=True
    )
    result["model_per_turn_percentile"] = result["model_tokens_per_turn"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_turn_percentile"] = result["num_turns_mean"].rank(
        method="average", pct=True, ascending=False
    )
    result["pd_selection_score"] = result[
        [
            "decode_pressure_percentile",
            "model_per_turn_percentile",
            "low_turn_percentile",
        ]
    ].mean(axis=1)

    ranked = result.sort_values(
        [
            "pd_selection_score",
            "decode_pressure_proxy",
            "model_tokens_per_turn",
            "num_turns_mean",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select agent tasks with high decode pressure for 1P+multi-D rollout"
    )
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="swe_rebench_pd64.parquet")
    parser.add_argument("--report", default="swe_rebench_pd_selection.csv")
    parser.add_argument("--num-samples", type=int, default=64)
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

    summary = rank_samples(summarize(records))
    selected_summary = summary.head(args.num_samples).copy()
    if len(selected_summary) < args.num_samples:
        raise ValueError(
            f"Only {len(selected_summary)} logged instances from the input parquet, "
            f"fewer than --num-samples={args.num_samples}"
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
    model_mean = float(records["model_tokens"].mean())
    print(f"ROLLOUT_SAMPLE records: {len(all_records)}")
    print(f"Matched records: {len(records)}")
    print(f"Ignored records outside input parquet: {ignored_records}")
    print(f"Ignored instances outside input parquet: {ignored_instances}")
    print(f"Logged sessions: {records['session_id'].nunique() if 'session_id' in records else len(records)}")
    print(f"Candidate instances: {len(summary)}")
    print(f"Input instances without logs: {len(missing_logged_ids)}")
    if missing_logged_ids:
        print(f"Missing instance_id values: {missing_logged_ids}")
    print(f"All-record prompt mean: {prompt_mean:.3f}")
    print(f"All-record response mean: {response_mean:.3f}")
    print(f"All-record model-token mean: {model_mean:.3f}")
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
            f"model_mean={row.model_tokens_mean:.0f} non_model_mean={row.non_model_tokens_mean:.0f} "
            f"model_per_turn={row.model_tokens_per_turn:.0f} "
            f"p_work_proxy={row.p_work_proxy_mean:.0f} "
            f"decode_pressure={row.decode_pressure_proxy:.3f} "
            f"pd_score={row.pd_selection_score:.3f} "
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
#   --num-samples 32

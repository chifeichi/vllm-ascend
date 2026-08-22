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
                    "final_context_tokens": prompt_tokens + response_tokens,
                    "p_work_proxy": p_work_proxy,
                    "decode_pressure_proxy": model_tokens / max(p_work_proxy, 1.0),
                    "model_tokens_per_turn": model_tokens / max(num_turns, 1.0),
                    "response_prompt_ratio": response_tokens / max(prompt_tokens, 1.0),
                    "model_prompt_ratio": model_tokens / max(prompt_tokens, 1.0),
                    "model_response_share": model_tokens / max(response_tokens, 1.0),
                }
            )

        sessions_frame = pd.DataFrame(session_rows)
        prompt_mean = float(sessions_frame["prompt_tokens"].mean())
        response_mean = float(sessions_frame["response_tokens"].mean())
        model_mean = float(sessions_frame["model_tokens"].mean())
        non_model_mean = float(sessions_frame["non_model_tokens"].mean())
        turns_mean = float(sessions_frame["num_turns"].mean())
        p_work_mean = float(sessions_frame["p_work_proxy"].mean())
        total_work = sessions_frame["p_work_proxy"] + sessions_frame["model_tokens"]
        model_per_turn = sessions_frame["model_tokens_per_turn"]
        model_per_turn_mean = float(model_per_turn.mean())
        model_per_turn_std = float(model_per_turn.std(ddof=0))
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
                "final_context_tokens_p75": float(
                    sessions_frame["final_context_tokens"].quantile(0.75)
                ),
                "final_context_tokens_max": float(sessions_frame["final_context_tokens"].max()),
                "model_tokens_p25": float(sessions_frame["model_tokens"].quantile(0.25)),
                "model_tokens_mean": model_mean,
                "non_model_tokens_mean": non_model_mean,
                "response_prompt_ratio_of_means": response_mean / max(prompt_mean, 1.0),
                "response_prompt_ratio_p25": float(
                    sessions_frame["response_prompt_ratio"].quantile(0.25)
                ),
                "model_prompt_ratio_of_means": model_mean / max(prompt_mean, 1.0),
                "model_prompt_ratio_p25": float(
                    sessions_frame["model_prompt_ratio"].quantile(0.25)
                ),
                "model_response_share_mean": float(sessions_frame["model_response_share"].mean()),
                "model_response_share_p25": float(
                    sessions_frame["model_response_share"].quantile(0.25)
                ),
                "num_turns_mean": turns_mean,
                "num_turns_p50": float(sessions_frame["num_turns"].median()),
                "num_turns_p75": float(sessions_frame["num_turns"].quantile(0.75)),
                "num_turns_p90": float(sessions_frame["num_turns"].quantile(0.90)),
                "model_tokens_per_turn": model_per_turn_mean,
                "model_tokens_per_turn_p25": float(model_per_turn.quantile(0.25)),
                "model_tokens_per_turn_std": model_per_turn_std,
                "model_tokens_per_turn_cv": (
                    model_per_turn_std / model_per_turn_mean
                    if model_per_turn_mean > 0
                    else float("inf")
                ),
                "p_work_proxy_mean": p_work_mean,
                "p_work_proxy_p75": float(sessions_frame["p_work_proxy"].quantile(0.75)),
                "p_work_proxy_max": float(sessions_frame["p_work_proxy"].max()),
                "total_work_proxy_mean": float(total_work.mean()),
                "total_work_proxy_max": float(total_work.max()),
                "decode_pressure_proxy": model_mean / max(p_work_mean, 1.0),
                "decode_pressure_proxy_p25": float(
                    sessions_frame["decode_pressure_proxy"].quantile(0.25)
                ),
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
    tail_quantile: float,
    cache_quantile: float,
    max_turns_quantile: float,
    rollout_n: int,
    min_model_prompt_ratio: float | None,
    max_turns_mean: float | None,
) -> pd.DataFrame:
    result = summary.copy()
    result["stable_decode_p_work_percentile"] = result["decode_pressure_proxy_p25"].rank(
        method="average", pct=True, ascending=True
    )
    result["stable_model_per_turn_percentile"] = result["model_tokens_per_turn_p25"].rank(
        method="average", pct=True, ascending=True
    )
    result["stable_model_tokens_percentile"] = result["model_tokens_p25"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_final_context_percentile"] = 1.0 - result["final_context_tokens_p75"].rank(
        method="average", pct=True, ascending=True
    )
    result["stable_rollouts_percentile"] = 1.0 - result["model_tokens_per_turn_cv"].rank(
        method="average", pct=True, ascending=True
    )

    # ROLLOUT_SAMPLE has token totals but no session/model/tool wall times.  This
    # score balances stable D-side work against estimated P-side recomputation
    # and prefix-cache footprint.  Cache/tail eligibility below remains a hard
    # guardrail; the score also continuously prefers smaller final contexts.
    result["pd_suitability_score"] = (
        0.40 * result["stable_decode_p_work_percentile"]
        + 0.25 * result["stable_model_per_turn_percentile"]
        + 0.15 * result["stable_model_tokens_percentile"]
        + 0.10 * result["low_final_context_percentile"]
        + 0.10 * result["stable_rollouts_percentile"]
    )
    result["pd_score_rank_all"] = result["pd_suitability_score"].rank(
        method="min", ascending=False
    ).astype(int)

    tail_cutoff = float(result["total_work_proxy_max"].quantile(tail_quantile))
    result["tail_work_cutoff"] = tail_cutoff
    result["tail_eligible"] = result["total_work_proxy_max"] <= tail_cutoff

    p_work_cutoff = float(result["p_work_proxy_p75"].quantile(cache_quantile))
    context_cutoff = float(result["final_context_tokens_p75"].quantile(cache_quantile))
    result["p_work_cache_cutoff"] = p_work_cutoff
    result["context_cache_cutoff"] = context_cutoff
    result["cache_eligible"] = (
        (result["p_work_proxy_p75"] <= p_work_cutoff)
        & (result["final_context_tokens_p75"] <= context_cutoff)
    )

    result["rollout_n_eligible"] = (
        (result["sessions"] >= rollout_n)
        & (result["sessions"] % rollout_n == 0)
        & (result["trajectory_records"] == result["sessions"])
    )
    result["ratio_eligible"] = (
        True
        if min_model_prompt_ratio is None
        else result["model_prompt_ratio_p25"] >= min_model_prompt_ratio
    )
    turns_cutoff = (
        float(result["num_turns_mean"].quantile(max_turns_quantile))
        if max_turns_mean is None
        else max_turns_mean
    )
    result["turns_cutoff"] = turns_cutoff
    result["turns_eligible"] = result["num_turns_mean"] <= turns_cutoff
    result["trajectory_eligible"] = (
        (result["exit_success_rate"] >= 1.0)
        & (result["normal_reason_rate"] >= 1.0)
    )
    result["selection_eligible"] = (
        result["tail_eligible"]
        & result["cache_eligible"]
        & result["rollout_n_eligible"]
        & result["ratio_eligible"]
        & result["turns_eligible"]
        & result["trajectory_eligible"]
    )

    sort_columns = [
        "pd_suitability_score",
        "decode_pressure_proxy_p25",
        "model_tokens_per_turn_p25",
        "model_tokens_p25",
        "final_context_tokens_p75",
    ]
    ascending = [False, False, False, False, True]
    eligible = result[result["selection_eligible"]].sort_values(
        sort_columns,
        ascending=ascending,
    )
    excluded = result[~result["selection_eligible"]].sort_values(
        sort_columns,
        ascending=ascending,
    )
    ranked = pd.concat([eligible, excluded], ignore_index=True)
    ranked["rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked.loc[ranked["selection_eligible"], "rank"] = range(1, len(eligible) + 1)
    return ranked


def _validate_tail_quantile(value: float) -> float:
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("tail quantile must be in (0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select stable, cache-safe agent tasks with sustained model generation "
            "for 1P+multi-D rollout"
        )
    )
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--score-input",
        help=(
            "Optional previously selected parquet to score against the current "
            "--input candidate population. Does not affect the new selection."
        ),
    )
    parser.add_argument("--output", default="swe_rebench_pd64.parquet")
    parser.add_argument("--report", default="swe_rebench_pd_selection.csv")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=4,
        help=(
            "Expected repeated rollouts per instance. Instances must have one or "
            "more complete groups and one trajectory per session (default: 4)."
        ),
    )
    parser.add_argument("--tail-quantile", type=_validate_tail_quantile, default=0.90)
    parser.add_argument(
        "--cache-quantile",
        type=_validate_tail_quantile,
        default=0.80,
        help=(
            "Keep instances whose P75 P-work and final-context proxies are no "
            "larger than this population quantile (default: 0.80)."
        ),
    )
    parser.add_argument(
        "--max-turns-quantile",
        type=_validate_tail_quantile,
        default=0.75,
        help=(
            "Automatic mean-turn cutoff when --max-turns-mean is omitted "
            "(default: population P75)."
        ),
    )
    parser.add_argument("--min-model-prompt-ratio", type=float)
    parser.add_argument("--max-turns-mean", type=float)
    args = parser.parse_args()

    if args.rollout_n <= 0:
        parser.error("--rollout-n must be greater than zero")

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
        args.tail_quantile,
        args.cache_quantile,
        args.max_turns_quantile,
        args.rollout_n,
        args.min_model_prompt_ratio,
        args.max_turns_mean,
    )

    scored_input_ids: set[str] = set()
    missing_scored_ids: list[str] = []
    if args.score_input:
        scored_dataset = pd.read_parquet(args.score_input)
        scored_ids = scored_dataset.apply(dataset_instance_id, axis=1)
        empty_scored_id_count = int((scored_ids == "").sum())
        if empty_scored_id_count:
            raise ValueError(
                f"Cannot resolve instance_id for {empty_scored_id_count} rows in "
                f"--score-input={args.score_input}"
            )
        scored_input_ids = set(scored_ids)
        missing_scored_ids = sorted(scored_input_ids - set(summary["instance_id"]))
    summary["in_score_input"] = summary["instance_id"].isin(scored_input_ids)

    selected_summary = summary[summary["selection_eligible"]].head(args.num_samples).copy()
    summary["selected"] = summary["rank"].isin(selected_summary["rank"])
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.report, index=False)
    if len(selected_summary) < args.num_samples:
        raise ValueError(
            f"Only {len(selected_summary)} instances satisfy the enabled validity/tail constraints, "
            f"fewer than --num-samples={args.num_samples}. Do not fill the remainder with weak "
            f"relative candidates; inspect {args.report} eligibility columns or expand the input pool."
        )

    rank = dict(zip(selected_summary["instance_id"], selected_summary["rank"], strict=True))
    selected = dataset[ids.isin(rank)].copy()
    selected["_selection_rank"] = ids[ids.isin(rank)].map(rank).to_numpy()
    selected = selected.sort_values("_selection_rank").drop(columns="_selection_rank")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)

    missing_logged_ids = sorted(input_ids - set(records["instance_id"]))
    prompt_mean = float(records["prompt_tokens"].mean())
    response_mean = float(records["response_tokens"].mean())
    model_mean = float(records["model_tokens"].mean())
    selected_ids = set(selected_summary["instance_id"])
    selected_records = records[records["instance_id"].isin(selected_ids)]
    selected_prompt_total = float(selected_records["prompt_tokens"].sum())
    selected_response_total = float(selected_records["response_tokens"].sum())
    selected_model_total = float(selected_records["model_tokens"].sum())
    selected_prompt_mean = float(selected_records["prompt_tokens"].mean())
    selected_response_mean = float(selected_records["response_tokens"].mean())
    selected_model_mean = float(selected_records["model_tokens"].mean())
    selected_response_prompt_ratio = selected_response_total / max(selected_prompt_total, 1.0)
    print(f"ROLLOUT_SAMPLE records: {len(all_records)}")
    print(f"Matched records: {len(records)}")
    print(f"Ignored records outside input parquet: {ignored_records}")
    print(f"Ignored instances outside input parquet: {ignored_instances}")
    print(f"Logged sessions: {records['session_id'].nunique() if 'session_id' in records else len(records)}")
    print(f"Candidate instances: {len(summary)}")
    print(
        f"Tail-eligible instances: {int(summary['tail_eligible'].sum())} "
        f"(P{args.tail_quantile * 100:g} total-work cutoff="
        f"{summary['tail_work_cutoff'].iloc[0]:.0f})"
    )
    print(
        f"rollout.n-eligible instances: {int(summary['rollout_n_eligible'].sum())} "
        f"(complete groups of n={args.rollout_n}, one trajectory per session)"
    )
    print(
        f"Cache-eligible instances: {int(summary['cache_eligible'].sum())} "
        f"(P{args.cache_quantile * 100:g} cutoffs: "
        f"P-work P75<={summary['p_work_cache_cutoff'].iloc[0]:.0f}, "
        f"final-context P75<={summary['context_cache_cutoff'].iloc[0]:.0f})"
    )
    if args.min_model_prompt_ratio is None:
        print("Model/prompt hard threshold: disabled")
    else:
        print(
            f"Ratio-eligible instances: {int(summary['ratio_eligible'].sum())} "
            f"(model/prompt P25 >= {args.min_model_prompt_ratio:.6f})"
        )
    turns_source = (
        f"population P{args.max_turns_quantile * 100:g}"
        if args.max_turns_mean is None
        else "--max-turns-mean"
    )
    print(
        f"Turn-eligible instances: {int(summary['turns_eligible'].sum())} "
        f"(turns mean <= {summary['turns_cutoff'].iloc[0]:g}, {turns_source})"
    )
    print(f"Fully eligible instances: {int(summary['selection_eligible'].sum())}")
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
    print(f"Selected ROLLOUT_SAMPLE records: {len(selected_records)}")
    print(f"Selected prompt total: {selected_prompt_total:.0f}")
    print(f"Selected response total: {selected_response_total:.0f}")
    print(f"Selected model-token total: {selected_model_total:.0f}")
    print(f"Selected prompt mean: {selected_prompt_mean:.3f}")
    print(f"Selected response mean: {selected_response_mean:.3f}")
    print(f"Selected model-token mean: {selected_model_mean:.3f}")
    print(
        "Selected aggregate response:prompt ratio: "
        f"{selected_response_prompt_ratio:.6f}:1"
    )
    if args.score_input:
        scored_summary = summary[summary["in_score_input"]].sort_values(
            ["pd_score_rank_all", "instance_id"], kind="stable"
        )
        overlap = int(scored_summary["selected"].sum())
        print(f"Score-input parquet: {args.score_input}")
        print(f"Score-input instance IDs: {len(scored_input_ids)}")
        print(f"Score-input IDs matched to current report: {len(scored_summary)}")
        print(f"Score-input IDs missing from current logs/input: {len(missing_scored_ids)}")
        if missing_scored_ids:
            print(f"Missing score-input instance_id values: {missing_scored_ids}")
        print(f"Score-input overlap with new selection: {overlap}/{len(scored_summary)}")
        if not scored_summary.empty:
            print(
                "Score-input pd_suitability_score "
                f"mean={scored_summary['pd_suitability_score'].mean():.6f} "
                f"min={scored_summary['pd_suitability_score'].min():.6f} "
                f"max={scored_summary['pd_suitability_score'].max():.6f}"
            )
            print("Score-input instances under current criteria:")
            for row in scored_summary.itertuples(index=False):
                current_rank = "-" if pd.isna(row.rank) else str(int(row.rank))
                failed = [
                    name.removesuffix("_eligible")
                    for name in (
                        "rollout_n_eligible",
                        "trajectory_eligible",
                        "tail_eligible",
                        "cache_eligible",
                        "turns_eligible",
                        "ratio_eligible",
                    )
                    if not getattr(row, name)
                ]
                print(
                    f"instance_id={row.instance_id} "
                    f"pd_suitability_score={row.pd_suitability_score:.6f} "
                    f"score_rank_all={row.pd_score_rank_all}/{len(summary)} "
                    f"model_p25={row.model_tokens_p25:.0f} "
                    f"model_per_turn_p25={row.model_tokens_per_turn_p25:.0f} "
                    f"p_work_p75={row.p_work_proxy_p75:.0f} "
                    f"decode_p_work_p25={row.decode_pressure_proxy_p25:.3f} "
                    f"context_p75={row.final_context_tokens_p75:.0f} "
                    f"eligible_rank={current_rank} "
                    f"selection_eligible={row.selection_eligible} "
                    f"selected_now={row.selected} "
                    f"failed={','.join(failed) if failed else '-'}"
                )
    print(f"Output parquet: {args.output}")
    print(f"Selection report: {args.report}")
    for row in selected_summary.itertuples(index=False):
        print(
            f"rank={row.rank} instance_id={row.instance_id} "
            f"sessions={row.sessions} trajectories={row.trajectory_records} "
            f"turns_mean={row.num_turns_mean:.1f} turns_p50={row.num_turns_p50:.1f} "
            f"prompt_mean={row.prompt_tokens_mean:.0f} response_mean={row.response_tokens_mean:.0f} "
            f"model_mean={row.model_tokens_mean:.0f} non_model_mean={row.non_model_tokens_mean:.0f} "
            f"model_per_turn_mean={row.model_tokens_per_turn:.0f} "
            f"model_per_turn_p25={row.model_tokens_per_turn_p25:.0f} "
            f"model_per_turn_cv={row.model_tokens_per_turn_cv:.3f} "
            f"p_work_proxy={row.p_work_proxy_mean:.0f} "
            f"p_work_p75={row.p_work_proxy_p75:.0f} "
            f"context_p75={row.final_context_tokens_p75:.0f} "
            f"total_work={row.total_work_proxy_mean:.0f} "
            f"decode_pressure={row.decode_pressure_proxy:.3f} "
            f"decode_pressure_p25={row.decode_pressure_proxy_p25:.3f} "
            f"pd_suitability_score={row.pd_suitability_score:.3f} "
            f"score_parts="
            f"ratio:{row.stable_decode_p_work_percentile:.3f},"
            f"per_turn:{row.stable_model_per_turn_percentile:.3f},"
            f"model:{row.stable_model_tokens_percentile:.3f},"
            f"low_context:{row.low_final_context_percentile:.3f},"
            f"stability:{row.stable_rollouts_percentile:.3f} "
            f"response_prompt_ratio_mean={row.response_prompt_ratio_of_means:.3f} "
            f"response_prompt_ratio_p25={row.response_prompt_ratio_p25:.3f} "
            f"model_prompt_ratio_mean={row.model_prompt_ratio_of_means:.3f} "
            f"model_prompt_ratio_p25={row.model_prompt_ratio_p25:.3f} "
            f"model_response_share_p25={row.model_response_share_p25:.3f} "
            f"exit_success_rate={row.exit_success_rate:.2f} resolved_rate={row.resolved_rate:.2f}"
        )


if __name__ == "__main__":
    main()


# python select_pd_benefit_samples.py \
#   --log 1.log \
#   --input swe_rebench_hard200.parquet \
#   --score-input old_swe_rebench_pd64.parquet \
#   --output swe_rebench_pd64.parquet \
#   --report swe_rebench_pd_selection.csv \
#   --num-samples 32 \
#   --rollout-n 4 \
#   --tail-quantile 0.90

#!/usr/bin/env python3

import argparse
import json
import math
import re
from itertools import combinations
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
        session_index_counts = None
        if session_key and "session_index" in group:
            session_metadata = group[["session_id", "session_index"]].drop_duplicates("session_id")
            session_metadata["session_index"] = pd.to_numeric(
                session_metadata["session_index"], errors="coerce"
            )
            session_metadata = session_metadata.dropna(subset=["session_index"])
            session_index_counts = {
                int(index): int(count)
                for index, count in session_metadata["session_index"].value_counts().items()
            }
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
        total_work_mean = float(total_work.mean())
        total_work_p50 = float(total_work.median())
        total_work_std = float(total_work.std(ddof=0))
        model_per_turn = sessions_frame["model_tokens_per_turn"]
        model_per_turn_mean = float(model_per_turn.mean())
        model_per_turn_std = float(model_per_turn.std(ddof=0))
        sessions = len(sessions_frame)
        rows.append(
            {
                "instance_id": str(instance_id),
                "sessions": sessions,
                "trajectory_records": len(group),
                "session_index_counts": session_index_counts,
                "prompt_tokens_p25": float(sessions_frame["prompt_tokens"].quantile(0.25)),
                "prompt_tokens_p50": float(sessions_frame["prompt_tokens"].median()),
                "prompt_tokens_mean": prompt_mean,
                "response_tokens_min": float(sessions_frame["response_tokens"].min()),
                "response_tokens_p25": float(sessions_frame["response_tokens"].quantile(0.25)),
                "response_tokens_p50": float(sessions_frame["response_tokens"].median()),
                "response_tokens_mean": response_mean,
                "response_tokens_max": float(sessions_frame["response_tokens"].max()),
                "response_tokens_tail_ratio": float(
                    sessions_frame["response_tokens"].max()
                    / max(sessions_frame["response_tokens"].median(), 1.0)
                ),
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
                "num_turns_max": float(sessions_frame["num_turns"].max()),
                "num_turns_tail_ratio": float(
                    sessions_frame["num_turns"].max()
                    / max(sessions_frame["num_turns"].median(), 1.0)
                ),
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
                "total_work_proxy_mean": total_work_mean,
                "total_work_proxy_p50": total_work_p50,
                "total_work_proxy_std": total_work_std,
                "total_work_proxy_cv": (
                    total_work_std / total_work_mean
                    if total_work_mean > 0
                    else float("inf")
                ),
                "total_work_proxy_max": float(total_work.max()),
                "total_work_proxy_tail_ratio": float(
                    total_work.max() / max(total_work_p50, 1.0)
                ),
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
    min_decode_pressure_quantile: float,
    min_decode_work_quantile: float,
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
    result["low_p_work_percentile"] = 1.0 - result["p_work_proxy_p75"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_total_work_percentile"] = 1.0 - result["total_work_proxy_p50"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_final_context_percentile"] = 1.0 - result["final_context_tokens_p75"].rank(
        method="average", pct=True, ascending=True
    )
    result["stable_rollouts_percentile"] = 1.0 - result["model_tokens_per_turn_cv"].rank(
        method="average", pct=True, ascending=True
    )

    # ROLLOUT_SAMPLE has token totals but no session/model/tool wall times.  This
    # Prefer D work that is efficient relative to P/cache cost, rather than
    # maximizing absolute D work. Cache/tail eligibility below remains a hard
    # guardrail, while P and total work are also continuous score penalties.
    result["pd_suitability_score"] = (
        0.35 * result["stable_decode_p_work_percentile"]
        + 0.10 * result["stable_model_per_turn_percentile"]
        + 0.10 * result["stable_model_tokens_percentile"]
        + 0.20 * result["low_p_work_percentile"]
        + 0.10 * result["low_total_work_percentile"]
        + 0.05 * result["low_final_context_percentile"]
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

    decode_pressure_cutoff = float(
        result["decode_pressure_proxy_p25"].quantile(min_decode_pressure_quantile)
    )
    result["decode_pressure_cutoff"] = decode_pressure_cutoff
    result["decode_pressure_eligible"] = (
        result["decode_pressure_proxy_p25"] >= decode_pressure_cutoff
    )

    decode_work_cutoff = float(
        result["model_tokens_p25"].quantile(min_decode_work_quantile)
    )
    result["decode_work_cutoff"] = decode_work_cutoff
    result["decode_work_eligible"] = result["model_tokens_p25"] >= decode_work_cutoff

    def has_complete_rollout_groups(row: pd.Series) -> bool:
        counts = row.get("session_index_counts")
        if isinstance(counts, dict) and counts:
            expected_indices = set(range(rollout_n))
            return (
                set(counts) == expected_indices
                and len(set(counts.values())) == 1
                and next(iter(counts.values())) > 0
            )
        sessions = int(row["sessions"])
        return sessions >= rollout_n and sessions % rollout_n == 0

    result["rollout_n_eligible"] = result.apply(has_complete_rollout_groups, axis=1)
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
        & result["decode_pressure_eligible"]
        & result["decode_work_eligible"]
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


def mark_stable_candidates(
    summary: pd.DataFrame,
    max_work_cv: float,
    max_work_tail_ratio: float,
    max_turns_tail_ratio: float,
    max_response_tail_ratio: float,
) -> pd.DataFrame:
    result = summary.copy()
    result["work_cv_eligible"] = result["total_work_proxy_cv"] <= max_work_cv
    result["work_tail_ratio_eligible"] = (
        result["total_work_proxy_tail_ratio"] <= max_work_tail_ratio
    )
    result["turns_tail_ratio_eligible"] = (
        result["num_turns_tail_ratio"] <= max_turns_tail_ratio
    )
    result["response_tail_ratio_eligible"] = (
        result["response_tokens_tail_ratio"] <= max_response_tail_ratio
    )
    result["stability_eligible"] = (
        result["work_cv_eligible"]
        & result["work_tail_ratio_eligible"]
        & result["turns_tail_ratio_eligible"]
        & result["response_tail_ratio_eligible"]
    )
    result["cohort_eligible"] = result["selection_eligible"] & result["stability_eligible"]
    return result


def select_similar_cohort(
    summary: pd.DataFrame,
    num_samples: int,
    pool_size: int,
    closeness_weight: float,
    reference_weight: float,
    reference_levels: dict[str, float] | None,
    max_cohort_cv: float,
    max_cohort_max_min: float,
    max_reference_ratio: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    workload_features = {
        "p_work_proxy_mean": 0.30,
        "model_tokens_mean": 0.25,
        "decode_pressure_proxy": 0.20,
        "total_work_proxy_mean": 0.25,
    }
    candidates = summary[summary["cohort_eligible"]].copy()
    if reference_levels is not None:
        candidates["_reference_distance"] = sum(
            workload_features[name]
            * (candidates[name].clip(lower=1e-12) / reference_levels[name])
            .map(math.log)
            .abs()
            for name in workload_features
        )
        candidates["_pool_priority"] = (
            candidates["pd_suitability_score"]
            - reference_weight * candidates["_reference_distance"]
        )
    else:
        candidates["_pool_priority"] = candidates["pd_suitability_score"]
    candidates = candidates.sort_values(
        ["_pool_priority", "decode_pressure_proxy_p25"],
        ascending=[False, False],
    )
    candidates = candidates.head(pool_size)
    if len(candidates) < num_samples:
        raise ValueError(
            f"Only {len(candidates)} stable, fully eligible instances are available "
            f"for a similar cohort of {num_samples}. Inspect stability eligibility "
            "columns in the report or relax the cohort tail thresholds."
        )

    combination_count = math.comb(len(candidates), num_samples)
    if combination_count > 1_000_000:
        raise ValueError(
            f"Similar-cohort search would evaluate {combination_count} combinations; "
            "reduce --cohort-pool-size or --num-samples."
        )

    best_indices: tuple[int, ...] | None = None
    best_key: tuple[float, float, float, float] | None = None
    best_stats: dict[str, float] = {}
    for indices in combinations(candidates.index.tolist(), num_samples):
        cohort = candidates.loc[list(indices)]
        cvs = {
            name: float(cohort[name].std(ddof=0) / max(cohort[name].mean(), 1e-12))
            for name in workload_features
        }
        max_min_ratios = {
            name: float(cohort[name].max() / max(cohort[name].min(), 1e-12))
            for name in workload_features
        }
        if any(value > max_cohort_cv for value in cvs.values()) or any(
            value > max_cohort_max_min for value in max_min_ratios.values()
        ):
            continue
        spreads = {
            name: max_min_ratios[name] - 1.0 for name in workload_features
        }
        closeness_penalty = sum(
            workload_features[name] * spreads[name]
            for name in workload_features
        )
        mean_score = float(cohort["pd_suitability_score"].mean())
        min_score = float(cohort["pd_suitability_score"].min())
        score_quality = 0.70 * mean_score + 0.30 * min_score
        cohort_levels = {
            "p_work_proxy_mean": float(cohort["p_work_proxy_mean"].mean()),
            "model_tokens_mean": float(cohort["model_tokens_mean"].mean()),
            "decode_pressure_proxy": float(
                cohort["model_tokens_mean"].sum()
                / max(float(cohort["p_work_proxy_mean"].sum()), 1e-12)
            ),
            "total_work_proxy_mean": float(cohort["total_work_proxy_mean"].mean()),
        }
        reference_penalty = 0.0
        if reference_levels is not None:
            reference_ratios = {
                name: max(
                    cohort_levels[name] / reference_levels[name],
                    reference_levels[name] / cohort_levels[name],
                )
                for name in workload_features
            }
            if any(value > max_reference_ratio for value in reference_ratios.values()):
                continue
            reference_penalty = sum(
                workload_features[name]
                * abs(math.log(cohort_levels[name] / reference_levels[name]))
                for name in workload_features
            )
        objective = (
            score_quality
            - closeness_weight * closeness_penalty
            - reference_weight * reference_penalty
        )
        key = (
            objective,
            -closeness_penalty,
            mean_score,
            min_score,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_indices = indices
            best_stats = {
                "objective": objective,
                "score_quality": score_quality,
                "mean_score": mean_score,
                "min_score": min_score,
                "closeness_penalty": closeness_penalty,
                "reference_penalty": reference_penalty,
                **{f"spread_{name}": value for name, value in spreads.items()},
                **{f"cv_{name}": value for name, value in cvs.items()},
                **{f"mean_{name}": value for name, value in cohort_levels.items()},
            }

    if best_indices is None:
        raise ValueError(
            "No cohort satisfies the batch-balance constraints: "
            f"CV<={max_cohort_cv:g} and max/min<={max_cohort_max_min:g} "
            "for P, D, D/P, and total"
            + (
                f", with aggregate/reference ratio<={max_reference_ratio:g}."
                if reference_levels is not None
                else "."
            )
            + " Increase --cohort-pool-size if possible, expand the input pool, "
            "or explicitly relax the limits."
        )
    selected = candidates.loc[list(best_indices)].sort_values(
        ["pd_suitability_score", "decode_pressure_proxy_p25"],
        ascending=[False, False],
    )
    return selected.copy(), best_stats


def _validate_tail_quantile(value: str | float) -> float:
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("tail quantile must be in (0, 1]")
    return number


def _validate_positive(value: str | float) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _validate_at_least_one(value: str | float) -> float:
    number = float(value)
    if number < 1.0:
        raise argparse.ArgumentTypeError("tail ratio must be at least 1")
    return number


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
        "--similar-cohort",
        action="store_true",
        help=(
            "Jointly select a stable group whose P, D, D/P, and total workloads "
            "are mutually balanced."
        ),
    )
    parser.add_argument(
        "--cohort-pool-size",
        type=int,
        default=40,
        help="Highest-scoring stable candidates considered by cohort search (default: 40).",
    )
    parser.add_argument(
        "--cohort-closeness-weight",
        type=_validate_positive,
        default=0.10,
        help="Penalty weight for workload spread inside the selected batch (default: 0.10).",
    )
    parser.add_argument(
        "--cohort-reference-weight",
        type=_validate_positive,
        default=0.35,
        help=(
            "Soft penalty for moving away from the aggregate workload of "
            "--score-input (default: 0.35; inactive without --score-input)."
        ),
    )
    parser.add_argument(
        "--max-cohort-cv",
        type=_validate_positive,
        default=0.30,
        help="Maximum batch-level CV for each of P, D, D/P, and total (default: 0.30).",
    )
    parser.add_argument(
        "--max-cohort-max-min",
        type=_validate_at_least_one,
        default=2.50,
        help=(
            "Maximum batch-level max/min ratio for each of P, D, D/P, and total "
            "(default: 2.50)."
        ),
    )
    parser.add_argument(
        "--max-cohort-reference-ratio",
        type=_validate_at_least_one,
        default=1.25,
        help=(
            "Maximum symmetric ratio between the selected batch aggregate and "
            "--score-input for P, D, D/P, and total (default: 1.25)."
        ),
    )
    parser.add_argument("--max-work-cv", type=_validate_positive, default=0.35)
    parser.add_argument(
        "--max-work-tail-ratio", type=_validate_at_least_one, default=1.50
    )
    parser.add_argument(
        "--max-turns-tail-ratio", type=_validate_at_least_one, default=1.50
    )
    parser.add_argument(
        "--max-response-tail-ratio", type=_validate_at_least_one, default=1.50
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=4,
        help=(
            "Expected repeated rollouts per instance. Instances must have one or "
            "more complete groups and one trajectory per session (default: 4)."
        ),
    )
    parser.add_argument("--tail-quantile", type=_validate_tail_quantile, default=0.85)
    parser.add_argument(
        "--cache-quantile",
        type=_validate_tail_quantile,
        default=0.70,
        help=(
            "Keep instances whose P75 P-work and final-context proxies are no "
            "larger than this population quantile (default: 0.70)."
        ),
    )
    parser.add_argument(
        "--min-decode-pressure-quantile",
        type=_validate_tail_quantile,
        default=0.75,
        help=(
            "Keep instances whose stable D/P proxy is at or above this population "
            "quantile (default: 0.75)."
        ),
    )
    parser.add_argument(
        "--min-decode-work-quantile",
        type=_validate_tail_quantile,
        default=0.50,
        help=(
            "Keep instances whose stable D work is at or above this population "
            "quantile (default: 0.50)."
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
    if args.num_samples <= 0:
        parser.error("--num-samples must be greater than zero")
    if args.cohort_pool_size <= 0:
        parser.error("--cohort-pool-size must be greater than zero")

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

    summary = mark_stable_candidates(
        rank_samples(
            summarize(records),
            args.tail_quantile,
            args.cache_quantile,
            args.min_decode_pressure_quantile,
            args.min_decode_work_quantile,
            args.max_turns_quantile,
            args.rollout_n,
            args.min_model_prompt_ratio,
            args.max_turns_mean,
        ),
        args.max_work_cv,
        args.max_work_tail_ratio,
        args.max_turns_tail_ratio,
        args.max_response_tail_ratio,
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

    cohort_stats: dict[str, float] | None = None
    selection_error: ValueError | None = None
    if args.similar_cohort:
        reference_levels: dict[str, float] | None = None
        if args.score_input:
            if missing_scored_ids:
                raise ValueError(
                    "Cannot use --score-input as a cohort reference because "
                    f"{len(missing_scored_ids)} reference instances are absent "
                    "from the current input/logs."
                )
            reference = summary[summary["in_score_input"]]
            if reference.empty:
                raise ValueError("--score-input contains no instances to use as a reference")
            reference_levels = {
                "p_work_proxy_mean": float(reference["p_work_proxy_mean"].mean()),
                "model_tokens_mean": float(reference["model_tokens_mean"].mean()),
                "decode_pressure_proxy": float(
                    reference["model_tokens_mean"].sum()
                    / max(float(reference["p_work_proxy_mean"].sum()), 1e-12)
                ),
                "total_work_proxy_mean": float(
                    reference["total_work_proxy_mean"].mean()
                ),
            }
        try:
            selected_summary, cohort_stats = select_similar_cohort(
                summary,
                args.num_samples,
                args.cohort_pool_size,
                args.cohort_closeness_weight,
                args.cohort_reference_weight,
                reference_levels,
                args.max_cohort_cv,
                args.max_cohort_max_min,
                args.max_cohort_reference_ratio,
            )
        except ValueError as exc:
            selected_summary = summary.iloc[0:0].copy()
            selection_error = exc
    else:
        selected_summary = summary[summary["selection_eligible"]].head(args.num_samples).copy()

    selected_summary["selection_rank"] = range(1, len(selected_summary) + 1)
    selection_rank = dict(
        zip(selected_summary["instance_id"], selected_summary["selection_rank"], strict=True)
    )
    summary["selected"] = summary["instance_id"].isin(selection_rank)
    summary["selection_rank"] = summary["instance_id"].map(selection_rank).astype("Int64")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.report, index=False)
    if selection_error is not None:
        raise selection_error
    if len(selected_summary) < args.num_samples:
        raise ValueError(
            f"Only {len(selected_summary)} instances satisfy the enabled validity/tail constraints, "
            f"fewer than --num-samples={args.num_samples}. Do not fill the remainder with weak "
            f"relative candidates; inspect {args.report} eligibility columns or expand the input pool."
        )

    selected = dataset[ids.isin(selection_rank)].copy()
    selected["_selection_rank"] = ids[ids.isin(selection_rank)].map(selection_rank).to_numpy()
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
        f"(complete session_index groups of n={args.rollout_n})"
    )
    print(
        f"Cache-eligible instances: {int(summary['cache_eligible'].sum())} "
        f"(P{args.cache_quantile * 100:g} cutoffs: "
        f"P-work P75<={summary['p_work_cache_cutoff'].iloc[0]:.0f}, "
        f"final-context P75<={summary['context_cache_cutoff'].iloc[0]:.0f})"
    )
    print(
        "Decode-pressure-eligible instances: "
        f"{int(summary['decode_pressure_eligible'].sum())} "
        f"(D/P P25 >= population P{args.min_decode_pressure_quantile * 100:g} "
        f"cutoff={summary['decode_pressure_cutoff'].iloc[0]:.3f})"
    )
    print(
        f"Decode-work-eligible instances: {int(summary['decode_work_eligible'].sum())} "
        f"(D P25 >= population P{args.min_decode_work_quantile * 100:g} "
        f"cutoff={summary['decode_work_cutoff'].iloc[0]:.0f})"
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
    print(
        f"Stability-eligible instances: {int(summary['stability_eligible'].sum())} "
        f"(work CV<={args.max_work_cv:g}, max/median: "
        f"work<={args.max_work_tail_ratio:g}, "
        f"turns<={args.max_turns_tail_ratio:g}, "
        f"response<={args.max_response_tail_ratio:g})"
    )
    if args.similar_cohort:
        print(
            f"Cohort-eligible instances: {int(summary['cohort_eligible'].sum())} "
            "(fully eligible and stable)"
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
    if cohort_stats is not None:
        print(
            "Similar-cohort objective: "
            f"{cohort_stats['objective']:.6f} "
            f"(score_quality={cohort_stats['score_quality']:.6f}, "
            f"mean_score={cohort_stats['mean_score']:.6f}, "
            f"min_score={cohort_stats['min_score']:.6f}, "
            f"reference_penalty={cohort_stats['reference_penalty']:.6f}, "
            f"balance_penalty={cohort_stats['closeness_penalty']:.6f})"
        )
        if reference_levels is not None:
            print(
                "Similar-cohort dynamic reference: "
                f"P={reference_levels['p_work_proxy_mean']:.0f}, "
                f"D={reference_levels['model_tokens_mean']:.0f}, "
                f"D/P={reference_levels['decode_pressure_proxy']:.3f}, "
                f"total={reference_levels['total_work_proxy_mean']:.0f} "
                f"(selected/reference ratio<={args.max_cohort_reference_ratio:g})"
            )
        print(
            "Similar-cohort workload means: "
            f"P={cohort_stats['mean_p_work_proxy_mean']:.0f}, "
            f"D={cohort_stats['mean_model_tokens_mean']:.0f}, "
            f"D/P={cohort_stats['mean_decode_pressure_proxy']:.3f}, "
            f"total={cohort_stats['mean_total_work_proxy_mean']:.0f}"
        )
        print(
            "Similar-cohort relative spreads: "
            f"P={cohort_stats['spread_p_work_proxy_mean']:.3f}, "
            f"D={cohort_stats['spread_model_tokens_mean']:.3f}, "
            f"D/P={cohort_stats['spread_decode_pressure_proxy']:.3f}, "
            f"total={cohort_stats['spread_total_work_proxy_mean']:.3f}"
        )
        print(
            "Similar-cohort CV: "
            f"P={cohort_stats['cv_p_work_proxy_mean']:.3f}, "
            f"D={cohort_stats['cv_model_tokens_mean']:.3f}, "
            f"D/P={cohort_stats['cv_decode_pressure_proxy']:.3f}, "
            f"total={cohort_stats['cv_total_work_proxy_mean']:.3f} "
            f"(hard limits: CV<={args.max_cohort_cv:g}, "
            f"max/min<={args.max_cohort_max_min:g})"
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
                        "decode_pressure_eligible",
                        "decode_work_eligible",
                        "turns_eligible",
                        "ratio_eligible",
                    )
                    if not getattr(row, name)
                ]
                if args.similar_cohort and not row.stability_eligible:
                    failed.append("stability")
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
            f"selection_rank={row.selection_rank} score_rank={row.rank} "
            f"instance_id={row.instance_id} "
            f"sessions={row.sessions} trajectories={row.trajectory_records} "
            f"turns_mean={row.num_turns_mean:.1f} turns_p50={row.num_turns_p50:.1f} "
            f"turns_max={row.num_turns_max:.1f} "
            f"turns_tail_ratio={row.num_turns_tail_ratio:.3f} "
            f"prompt_mean={row.prompt_tokens_mean:.0f} response_mean={row.response_tokens_mean:.0f} "
            f"response_tail_ratio={row.response_tokens_tail_ratio:.3f} "
            f"model_mean={row.model_tokens_mean:.0f} non_model_mean={row.non_model_tokens_mean:.0f} "
            f"model_per_turn_mean={row.model_tokens_per_turn:.0f} "
            f"model_per_turn_p25={row.model_tokens_per_turn_p25:.0f} "
            f"model_per_turn_cv={row.model_tokens_per_turn_cv:.3f} "
            f"p_work_proxy={row.p_work_proxy_mean:.0f} "
            f"p_work_p75={row.p_work_proxy_p75:.0f} "
            f"context_p75={row.final_context_tokens_p75:.0f} "
            f"total_work={row.total_work_proxy_mean:.0f} "
            f"total_work_p50={row.total_work_proxy_p50:.0f} "
            f"total_work_cv={row.total_work_proxy_cv:.3f} "
            f"total_work_tail_ratio={row.total_work_proxy_tail_ratio:.3f} "
            f"decode_pressure={row.decode_pressure_proxy:.3f} "
            f"decode_pressure_p25={row.decode_pressure_proxy_p25:.3f} "
            f"pd_suitability_score={row.pd_suitability_score:.3f} "
            f"score_parts="
            f"ratio:{row.stable_decode_p_work_percentile:.3f},"
            f"per_turn:{row.stable_model_per_turn_percentile:.3f},"
            f"model:{row.stable_model_tokens_percentile:.3f},"
            f"low_p:{row.low_p_work_percentile:.3f},"
            f"low_total:{row.low_total_work_percentile:.3f},"
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
#   --tail-quantile 0.85 \
#   --cache-quantile 0.70 \
#   --min-decode-pressure-quantile 0.75 \
#   --min-decode-work-quantile 0.50 \
#   --similar-cohort

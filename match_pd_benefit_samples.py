#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "decode_pressure_proxy_p25": 0.30,
    "model_tokens_per_turn_p25": 0.20,
    "model_tokens_p25": 0.15,
    "p_work_proxy_p75": 0.10,
    "final_context_tokens_p75": 0.10,
    "num_turns_p50": 0.10,
    "total_work_proxy_cv": 0.05,
}


def parse_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes"}
    )


def load_reference(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "selected" in frame:
        selected = parse_bool(frame["selected"])
        if selected.any():
            frame = frame[selected].copy()
    return frame


def load_candidates(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("rollout_n_eligible", "trajectory_eligible"):
        if column in frame:
            frame = frame[parse_bool(frame[column])].copy()
    return frame


def validate_columns(reference: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    required = {"instance_id", *FEATURES}
    missing_reference = sorted(required - set(reference.columns))
    missing_candidates = sorted(required - set(candidates.columns))
    if missing_reference:
        raise ValueError(f"Reference CSV is missing columns: {missing_reference}")
    if missing_candidates:
        raise ValueError(f"Candidate CSV is missing columns: {missing_candidates}")

    usable = []
    for name in FEATURES:
        reference[name] = pd.to_numeric(reference[name], errors="coerce")
        candidates[name] = pd.to_numeric(candidates[name], errors="coerce")
        if reference[name].notna().any() and candidates[name].notna().any():
            usable.append(name)
    if not usable:
        raise ValueError("No matching numeric feature columns contain usable values")
    return usable


def transformed(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame[features].to_numpy(dtype=float)
    return np.log1p(np.maximum(values, 0.0))


def normalized_features(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    reference_values = transformed(reference, features)
    candidate_values = transformed(candidates, features)
    reference_center = np.nanmedian(reference_values, axis=0)
    reference_q25 = np.nanquantile(reference_values, 0.25, axis=0)
    reference_q75 = np.nanquantile(reference_values, 0.75, axis=0)
    combined_std = np.nanstd(np.vstack([reference_values, candidate_values]), axis=0)
    scale = np.maximum(reference_q75 - reference_q25, combined_std * 0.25)
    scale = np.maximum(scale, 0.05)

    reference_values = np.where(np.isnan(reference_values), reference_center, reference_values)
    candidate_values = np.where(np.isnan(candidate_values), reference_center, candidate_values)
    return (
        (reference_values - reference_center) / scale,
        (candidate_values - reference_center) / scale,
    )


def choose_reference_targets(reference: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    if num_samples > len(reference):
        raise ValueError(
            f"--num-samples={num_samples} exceeds reference rows={len(reference)}"
        )
    if num_samples == len(reference):
        return reference.copy()
    ordered = reference.sort_values(
        ["decode_pressure_proxy_p25", "model_tokens_per_turn_p25"],
        kind="stable",
    )
    positions = np.linspace(0, len(ordered) - 1, num_samples).round().astype(int)
    return ordered.iloc[positions].copy()


def assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        remaining_rows = set(range(cost.shape[0]))
        remaining_columns = set(range(cost.shape[1]))
        pairs = []
        while remaining_rows:
            row, column = min(
                (
                    (row, column)
                    for row in remaining_rows
                    for column in remaining_columns
                ),
                key=lambda pair: cost[pair],
            )
            pairs.append((row, column))
            remaining_rows.remove(row)
            remaining_columns.remove(column)
        rows, columns = zip(*pairs, strict=True)
        return np.asarray(rows), np.asarray(columns)


def distribution_rows(
    reference: pd.DataFrame,
    selected: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for name in features:
        for statistic, function in (
            ("mean", pd.Series.mean),
            ("p25", lambda series: series.quantile(0.25)),
            ("p50", pd.Series.median),
            ("p75", lambda series: series.quantile(0.75)),
        ):
            old_value = float(function(reference[name]))
            new_value = float(function(selected[name]))
            rows.append(
                {
                    "feature": name,
                    "statistic": statistic,
                    "reference": old_value,
                    "selected": new_value,
                    "relative_error": abs(new_value - old_value) / max(abs(old_value), 1.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a new PD-benefit cohort matching an older report CSV"
    )
    parser.add_argument("--reference", required=True, help="Old well-performing report CSV")
    parser.add_argument("--candidates", required=True, help="New candidate report CSV")
    parser.add_argument("--output", default="matched_pd_selection.csv")
    parser.add_argument("--comparison", default="matched_pd_distribution.csv")
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    candidates = load_candidates(args.candidates)
    if reference.empty:
        raise ValueError("Reference CSV contains no target rows")
    if candidates.empty:
        raise ValueError("Candidate CSV contains no valid candidates")

    features = validate_columns(reference, candidates)
    num_samples = len(reference) if args.num_samples is None else args.num_samples
    if num_samples <= 0:
        parser.error("--num-samples must be greater than zero")
    if num_samples > len(candidates):
        raise ValueError(
            f"--num-samples={num_samples} exceeds valid candidates={len(candidates)}"
        )

    targets = choose_reference_targets(reference, num_samples)
    reference_values, candidate_values = normalized_features(targets, candidates, features)
    weights = np.asarray([FEATURES[name] for name in features], dtype=float)
    weights /= weights.sum()
    cost = np.sqrt(
        np.sum(
            weights[None, None, :]
            * (reference_values[:, None, :] - candidate_values[None, :, :]) ** 2,
            axis=2,
        )
    )
    target_rows, candidate_rows = assignment(cost)

    selected = candidates.iloc[candidate_rows].copy()
    selected["matched_reference_instance_id"] = targets.iloc[target_rows][
        "instance_id"
    ].astype(str).to_numpy()
    selected["reference_match_distance"] = cost[target_rows, candidate_rows]
    selected = selected.sort_values("reference_match_distance", kind="stable").reset_index(drop=True)
    selected.insert(0, "reference_match_rank", range(1, len(selected) + 1))

    comparison = distribution_rows(targets, selected, features)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    comparison.to_csv(args.comparison, index=False)

    print(f"Reference rows: {len(reference)}")
    print(f"Valid candidate rows: {len(candidates)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Mean reference-match distance: {selected['reference_match_distance'].mean():.6f}")
    print(f"Selected CSV: {args.output}")
    print(f"Distribution comparison CSV: {args.comparison}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

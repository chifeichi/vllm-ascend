#!/usr/bin/env python3

"""Print a compact P/D workload summary for a selected batch report."""

import argparse
import math

import pandas as pd


METRICS = {
    "P": "p_work_proxy_mean",
    "D": "model_tokens_mean",
    "D/P": "decode_pressure_proxy",
    "total": "total_work_proxy_mean",
}


def selected_rows(frame: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    selected = frame
    if "selected" in frame:
        mask = frame["selected"].map(
            lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
        )
        if mask.any():
            selected = frame[mask]
    elif "selection_rank" in frame and frame["selection_rank"].notna().any():
        selected = frame[frame["selection_rank"].notna()].sort_values("selection_rank")

    if top_k is not None:
        rank_column = next(
            (name for name in ("selection_rank", "rank", "pd_score_rank_all") if name in selected),
            None,
        )
        if rank_column is not None:
            selected = selected.sort_values(rank_column, na_position="last")
        selected = selected.head(top_k)
    return selected.copy()


def coefficient_of_variation(values: pd.Series) -> float:
    mean = float(values.mean())
    return float(values.std(ddof=0) / mean) if mean > 0 else float("nan")


def max_min_ratio(values: pd.Series) -> float:
    minimum = float(values.min())
    return float(values.max() / minimum) if minimum > 0 else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize P/D load and cross-task balance from a PD selection CSV."
    )
    parser.add_argument("csv", help="CSV report produced by select_pd_benefit_samples.py")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Use only the first K selected/ranked rows; defaults to every selected row.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    batch = selected_rows(frame, args.top_k)
    missing = [column for column in METRICS.values() if column not in batch]
    if missing:
        parser.error(f"CSV is missing required columns: {missing}")
    if batch.empty:
        parser.error("No selected rows found")

    values = {
        label: pd.to_numeric(batch[column], errors="coerce").dropna()
        for label, column in METRICS.items()
    }
    if any(series.empty for series in values.values()):
        parser.error("Selected rows contain no numeric P/D workload values")

    aggregate_ratio = values["D"].sum() / max(values["P"].sum(), 1.0)
    print(f"batch={len(batch)}")
    print(
        "level_mean "
        f"P={values['P'].mean():.0f} D={values['D'].mean():.0f} "
        f"D/P={aggregate_ratio:.3f} total={values['total'].mean():.0f}"
    )
    print(
        "balance_cv "
        + " ".join(
            f"{label}={fmt(coefficient_of_variation(series))}"
            for label, series in values.items()
        )
    )
    print(
        "balance_max/min "
        + " ".join(f"{label}={fmt(max_min_ratio(series))}" for label, series in values.items())
    )

    tail_columns = {
        "work": "total_work_proxy_tail_ratio",
        "response": "response_tokens_tail_ratio",
        "turns": "num_turns_tail_ratio",
    }
    available_tails = {
        label: pd.to_numeric(batch[column], errors="coerce").dropna()
        for label, column in tail_columns.items()
        if column in batch
    }
    if available_tails:
        print(
            "rollout_tail_mean "
            + " ".join(f"{label}={fmt(series.mean())}" for label, series in available_tails.items())
        )


if __name__ == "__main__":
    main()

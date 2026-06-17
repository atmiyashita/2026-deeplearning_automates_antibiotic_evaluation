#!/usr/bin/env python3
"""Summarize run-level raw survival-curve fluctuation metrics.

This script is intended for two or more independent runs. Each input CSV should
contain a time column and one or more raw survival-ratio columns.

Example
-------
python runlevel_fluctuation_summary_public.py \
  --run-csvs run1.csv,run2.csv \
  --run-labels batch1,batch2 \
  --survival-cols PAO1,PAO1_meropenem,saline \
  --output-dir outputs/fluctuation_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw survival-curve fluctuation metrics by run.")
    parser.add_argument("--run-csvs", required=True, help="Comma-separated CSV files, one per run.")
    parser.add_argument("--run-labels", default="", help="Optional comma-separated run labels.")
    parser.add_argument("--survival-cols", required=True, help="Comma-separated raw survival columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_column(path: Path, column: str) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [max(0.0, min(1.0, float(row[column]))) for row in rows if row.get(column, "") != ""]


def metrics(values: list[float]) -> dict[str, float]:
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    positive = [d for d in diffs if d > 0]
    return {
        "n_points": float(len(values)),
        "total_absolute_change": sum(abs(d) for d in diffs),
        "upward_fluctuation_sum": sum(positive),
        "max_upward_fluctuation": max(positive) if positive else 0.0,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_csvs = [Path(x) for x in split_csv_arg(args.run_csvs)]
    run_labels = split_csv_arg(args.run_labels) if args.run_labels else [f"run{i+1}" for i in range(len(run_csvs))]
    survival_cols = split_csv_arg(args.survival_cols)
    if len(run_labels) != len(run_csvs):
        raise ValueError("--run-labels must match --run-csvs length")

    long_rows: list[dict[str, str]] = []
    for run_path, run_label in zip(run_csvs, run_labels):
        for col in survival_cols:
            vals = read_column(run_path, col)
            m = metrics(vals)
            for metric, value in m.items():
                long_rows.append({"run": run_label, "group": col, "metric": metric, "value": f"{value:.6g}"})

    long_path = args.output_dir / "runlevel_fluctuation_metrics_long.csv"
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "group", "metric", "value"])
        writer.writeheader(); writer.writerows(long_rows)

    summary_rows: list[dict[str, str]] = []
    for col in survival_cols:
        for metric in sorted({row["metric"] for row in long_rows}):
            values = [float(row["value"]) for row in long_rows if row["group"] == col and row["metric"] == metric]
            mean, sd = mean_sd(values)
            summary_rows.append({"group": col, "metric": metric, "mean": f"{mean:.6g}", "sd": f"{sd:.6g}", "n_runs": str(len(values))})
    summary_path = args.output_dir / "runlevel_fluctuation_metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "metric", "mean", "sd", "n_runs"])
        writer.writeheader(); writer.writerows(summary_rows)

    metric_to_plot = "upward_fluctuation_sum"
    rows = [row for row in summary_rows if row["metric"] == metric_to_plot]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(rows)))
    means = [float(row["mean"]) for row in rows]
    sds = [float(row["sd"]) for row in rows]
    labels = [row["group"] for row in rows]
    ax.bar(x, means, yerr=sds, capsize=4)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(metric_to_plot)
    ax.set_title("Run-level raw survival fluctuation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_png = args.output_dir / "runlevel_fluctuation_summary.png"
    out_pdf = args.output_dir / "runlevel_fluctuation_summary.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "runlevel_fluctuation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"run_csvs": [str(x) for x in run_csvs], "run_labels": run_labels,
                   "survival_cols": survival_cols, "outputs": [str(long_path), str(summary_path), str(out_png), str(out_pdf)]},
                  handle, ensure_ascii=False, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

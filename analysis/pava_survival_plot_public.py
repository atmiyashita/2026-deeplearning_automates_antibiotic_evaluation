#!/usr/bin/env python3
"""Apply PAVA monotonic correction and plot survival curves from a CSV table.

Input CSV format
----------------
The CSV must contain one time column and one or more survival-ratio columns.
Survival values should be proportions between 0 and 1. Example:

    elapsed_hours,PAO1,PAO1_meropenem,saline
    0.0,1.0,1.0,1.0
    0.083,0.98,1.0,1.0

Example
-------
python pava_survival_plot_public.py \
  --input-csv survival_input.csv \
  --time-col elapsed_hours \
  --survival-cols PAO1,PAO1_meropenem,saline \
  --labels "P. aeruginosa,P. aeruginosa + meropenem,saline" \
  --output-dir outputs/survival_plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply PAVA to survival ratios and draw raw/PAVA survival curves."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--time-col", default="elapsed_hours")
    parser.add_argument(
        "--survival-cols",
        required=True,
        help="Comma-separated survival-ratio columns, e.g. PAO1,PAO1_meropenem,saline.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated display labels. Defaults to survival column names.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Detection-based survival curves")
    parser.add_argument("--x-label", default="Elapsed time (h)")
    parser.add_argument("--y-label", default="Survival ratio")
    return parser.parse_args()


def pava_nonincreasing(values: Iterable[float]) -> List[float]:
    """Pool adjacent violators algorithm for a non-increasing sequence."""
    y = [-float(v) for v in values]
    blocks: List[list[float]] = []  # [start, end, mean]
    for idx, value in enumerate(y):
        blocks.append([idx, idx, value])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            start1, end1, mean1 = blocks[-2]
            start2, end2, mean2 = blocks[-1]
            n1 = end1 - start1 + 1
            n2 = end2 - start2 + 1
            pooled_mean = (mean1 * n1 + mean2 * n2) / (n1 + n2)
            blocks[-2:] = [[start1, end2, pooled_mean]]

    corrected = [0.0] * len(y)
    for start, end, mean in blocks:
        for idx in range(int(start), int(end) + 1):
            corrected[idx] = max(0.0, min(1.0, -mean))
    return corrected


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.input_csv)
    survival_cols = split_csv_arg(args.survival_cols)
    labels = split_csv_arg(args.labels) if args.labels else survival_cols
    if len(labels) != len(survival_cols):
        raise ValueError("--labels must contain the same number of entries as --survival-cols")

    time_values = [float(row[args.time_col]) for row in rows]

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    summary: dict[str, dict[str, list[float]]] = {}

    for col, label in zip(survival_cols, labels):
        raw = [max(0.0, min(1.0, float(row[col]))) for row in rows]
        pava = pava_nonincreasing(raw)
        summary[col] = {"raw": raw, "pava": pava}
        ax.plot(time_values, raw, linewidth=1.0, alpha=0.35, label=f"{label} raw")
        ax.step(time_values, pava, where="post", linewidth=2.2, label=f"{label} PAVA")

    ax.set_title(args.title)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_png = args.output_dir / "survival_pava.png"
    out_pdf = args.output_dir / "survival_pava.pdf"
    out_svg = args.output_dir / "survival_pava.svg"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "survival_pava_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_csv": str(args.input_csv),
                "time_col": args.time_col,
                "survival_cols": survival_cols,
                "labels": labels,
                "outputs": [str(out_png), str(out_pdf), str(out_svg)],
                "series": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot survival curves with Greenwood 95% CI and pairwise log-rank tests.

Input CSV format
----------------
The CSV must contain one time column and one or more survival-proportion columns.
Survival values should be proportions between 0 and 1.

Example
-------
python km_greenwood_logrank_public.py \
  --input-csv survival_input.csv \
  --time-col elapsed_hours \
  --survival-cols PAO1,PAO1_meropenem,saline \
  --labels "P. aeruginosa,P. aeruginosa + meropenem,saline" \
  --n-total 30 \
  --output-dir outputs/km_plot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Kaplan-Meier-style survival curves with Greenwood CI and log-rank tests."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--time-col", default="elapsed_hours")
    parser.add_argument("--survival-cols", required=True, help="Comma-separated survival columns.")
    parser.add_argument("--labels", default="", help="Optional comma-separated display labels.")
    parser.add_argument("--n-total", type=float, required=True, help="Number of animals per group.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Survival curves")
    parser.add_argument("--x-label", default="Elapsed time (h)")
    parser.add_argument("--y-label", default="Kaplan-Meier survival probability")
    return parser.parse_args()


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def enforce_nonincreasing(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    current = 1.0
    for idx, value in enumerate(values):
        x = max(0.0, min(1.0, float(value)))
        current = x if idx == 0 else min(current, x)
        out.append(current)
    return out


def normalize_to_one(values: list[float]) -> list[float]:
    if not values or values[0] <= 0:
        return values
    return [max(0.0, min(1.0, v / values[0])) for v in values]


def greenwood_ci95(survival: list[float], n_total: float) -> tuple[list[float], list[float], list[float]]:
    z = 1.959963984540054
    cumulative = 0.0
    se: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for idx, s_now in enumerate(survival):
        s_now = max(0.0, min(1.0, s_now))
        if idx > 0:
            s_prev = max(0.0, min(1.0, survival[idx - 1]))
            n_i = n_total * s_prev
            d_i = max(0.0, n_total * (s_prev - s_now))
            if d_i > 0 and n_i > 0 and (n_i - d_i) > 0:
                cumulative += d_i / (n_i * (n_i - d_i))
        se_i = s_now * math.sqrt(max(0.0, cumulative))
        se.append(se_i)
        lo.append(max(0.0, s_now - z * se_i))
        hi.append(min(1.0, s_now + z * se_i))
    return se, lo, hi


def logrank_pvalue(s1: list[float], s2: list[float], n_total: float) -> dict[str, float]:
    observed_1 = 0.0
    expected_1 = 0.0
    variance_1 = 0.0
    for idx in range(1, min(len(s1), len(s2))):
        s1_prev, s1_now = s1[idx - 1], s1[idx]
        s2_prev, s2_now = s2[idx - 1], s2[idx]
        n1 = n_total * max(0.0, s1_prev)
        n2 = n_total * max(0.0, s2_prev)
        d1 = max(0.0, n_total * (s1_prev - s1_now))
        d2 = max(0.0, n_total * (s2_prev - s2_now))
        n = n1 + n2
        d = d1 + d2
        if n <= 0 or d <= 0:
            continue
        observed_1 += d1
        expected_1 += d * n1 / n
        if n > 1.0 and (n - d) > 0:
            variance_1 += (n1 * n2 * d * (n - d)) / (n * n * (n - 1.0))
    if variance_1 <= 0:
        return {"chi2": float("nan"), "p": float("nan")}
    chi2 = (observed_1 - expected_1) ** 2 / variance_1
    p = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return {"chi2": chi2, "p": p}


def fmt_p(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_csv)
    survival_cols = split_csv_arg(args.survival_cols)
    labels = split_csv_arg(args.labels) if args.labels else survival_cols
    if len(labels) != len(survival_cols):
        raise ValueError("--labels must match --survival-cols length")

    x = [float(row[args.time_col]) for row in rows]
    series: dict[str, list[float]] = {}
    ci_summary: dict[str, dict[str, list[float]]] = {}
    for col in survival_cols:
        y = normalize_to_one(enforce_nonincreasing(float(row[col]) for row in rows))
        series[col] = y
        se, lo, hi = greenwood_ci95(y, args.n_total)
        ci_summary[col] = {"se": se, "ci95_low": lo, "ci95_high": hi}

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label in zip(survival_cols, labels):
        ax.fill_between(x, ci_summary[col]["ci95_low"], ci_summary[col]["ci95_high"], step="post", alpha=0.18)
        ax.step(x, series[col], where="post", linewidth=2.2, label=f"{label} (n={int(args.n_total)})")
    ax.set_title(args.title)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    logrank: dict[str, dict[str, float]] = {}
    text_lines = ["log-rank"]
    for a, b in itertools.combinations(survival_cols, 2):
        result = logrank_pvalue(series[a], series[b], args.n_total)
        logrank[f"{a}_vs_{b}"] = result
        text_lines.append(f"{a} vs {b}: p={fmt_p(result['p'])}")
    ax.text(0.02, 0.60, "\n".join(text_lines), transform=ax.transAxes, fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    fig.tight_layout()

    out_png = args.output_dir / "survival_km_greenwood_logrank.png"
    out_pdf = args.output_dir / "survival_km_greenwood_logrank.pdf"
    out_svg = args.output_dir / "survival_km_greenwood_logrank.svg"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "survival_km_greenwood_logrank_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"input_csv": str(args.input_csv), "n_total": args.n_total, "logrank": logrank,
                   "ci_method": "Greenwood normal approximation", "outputs": [str(out_png), str(out_pdf), str(out_svg)]},
                  handle, ensure_ascii=False, indent=2)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

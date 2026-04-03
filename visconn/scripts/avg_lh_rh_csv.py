#!/usr/bin/env python3
"""
avg_lh_rh_csv.py — Compute bilateral (left + right hemisphere) averages
from two VOF coverage CSV files, write a combined CSV, and generate
a matching bar-plot figure.

Usage:
    python3 avg_lh_rh_csv.py \
        --lh   lh_vof_ventral_coverage.csv \
        --rh   rh_vof_ventral_coverage.csv \
        --out  bilateral_vof_ventral_coverage.csv \
        [--group ventral]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils import save_figure, nature_style_plot


NUMERIC_COLS = ["n_roi_vox", "n_covered_vox", "coverage_pct", "mean_tdi"]


def _strip_hemi_prefix(roi_name: str) -> str:
    return re.sub(r"^(?:lh|rh)[._]", "", roi_name, flags=re.IGNORECASE)


def _pretty_roi_label(roi_name: str) -> str:
    s = roi_name
    s = re.sub(r"\.bin(?:\.nii(?:\.gz)?)?$", "", s, flags=re.IGNORECASE)
    return s


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _make_plot(
    out_prefix: Path,
    rows: List[dict],
    metric: str = "coverage_pct",
    ylabel: str = "VOF map coverage (%)",
    fontsize: float = 16,
) -> None:
    rois = [_pretty_roi_label(r["roi"]) for r in rows]
    vals = np.array([float(r[metric]) if r[metric] != "" else np.nan for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(rois))
    ax.bar(x, vals, width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
    ymax = max(1.0, float(np.ceil(ymax * 10) / 10.0))
    ymin = 0.0
    ymid = round((ymin + ymax) / 2.0, 1)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks([ymin, ymid, ymax])
    ax.set_yticklabels([f"{v:.1f}" for v in [ymin, ymid, ymax]])

    nature_style_plot(ax, ymin=ymin, ymax=ymax, fontsize=fontsize, n_yticks=3)

    plt.tight_layout()
    save_figure(out_prefix, dpi=300)
    plt.close()

    print(f"[avg_lh_rh_csv] Figure written → {out_prefix}.png/.pdf/.svg")


def avg_lh_rh_csv(
    lh_csv: Path,
    rh_csv: Path,
    out_csv: Path,
    group: Optional[str] = None,
    make_figure: bool = True,
) -> Path:
    lh_rows = _read_csv(lh_csv)
    rh_rows = _read_csv(rh_csv)

    lh_map: Dict[str, dict] = {_strip_hemi_prefix(r["roi"]): r for r in lh_rows}
    rh_map: Dict[str, dict] = {_strip_hemi_prefix(r["roi"]): r for r in rh_rows}

    all_keys = sorted(set(lh_map) | set(rh_map))

    out_rows = []
    for key in all_keys:
        lh_row = lh_map.get(key)
        rh_row = rh_map.get(key)

        row_out: dict = {
            "hemisphere": "bilateral",
            "group": group or (lh_row or rh_row or {}).get("group", ""),
            "roi": key,
        }

        for col in NUMERIC_COLS:
            vals = []
            for r in (lh_row, rh_row):
                if r is not None:
                    try:
                        vals.append(float(r[col]))
                    except (KeyError, ValueError, TypeError):
                        pass
            row_out[col] = round(sum(vals) / len(vals), 6) if vals else ""

        for prefix, src_row in (("lh_", lh_row), ("rh_", rh_row)):
            for col in NUMERIC_COLS:
                if src_row is not None:
                    try:
                        row_out[prefix + col] = round(float(src_row[col]), 6)
                    except (KeyError, ValueError, TypeError):
                        row_out[prefix + col] = ""
                else:
                    row_out[prefix + col] = ""

        out_rows.append(row_out)

    fieldnames = (
        ["hemisphere", "group", "roi"]
        + NUMERIC_COLS
        + [p + c for p in ("lh_", "rh_") for c in NUMERIC_COLS]
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[avg_lh_rh_csv] Written {len(out_rows)} rows → {out_csv}")

    if make_figure:
        out_prefix = out_csv.with_suffix("")
        if out_prefix.suffix == ".nii":
            out_prefix = out_prefix.with_suffix("")
        _make_plot(out_prefix, out_rows, metric="coverage_pct")

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute bilateral averages from LH and RH VOF coverage CSVs."
    )
    ap.add_argument("--lh", required=True, type=Path,
                    help="Left-hemisphere CSV")
    ap.add_argument("--rh", required=True, type=Path,
                    help="Right-hemisphere CSV")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output bilateral CSV")
    ap.add_argument("--group", default=None,
                    help="ROI group label (e.g. ventral, dorsal, lo)")
    ap.add_argument("--no-figure", action="store_true",
                    help="Do not create bilateral figure")
    args = ap.parse_args()

    for p in [args.lh, args.rh]:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    avg_lh_rh_csv(
        args.lh,
        args.rh,
        args.out,
        args.group,
        make_figure=not args.no_figure,
    )


if __name__ == "__main__":
    main()

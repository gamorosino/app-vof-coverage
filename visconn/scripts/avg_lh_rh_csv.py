#!/usr/bin/env python3
"""
avg_lh_rh_csv.py — Compute bilateral (left + right hemisphere) averages
from two VOF coverage CSV files and write a combined summary.

The script reads one CSV for the left hemisphere and one for the right hemisphere
(both produced by vof_map_coverage.py), matches rows by ROI base-name (stripping
the hemisphere prefix), and outputs a new CSV with per-ROI bilateral averages.

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


# Columns to average numerically
NUMERIC_COLS = ["n_roi_vox", "n_covered_vox", "coverage_pct", "mean_tdi"]


def _strip_hemi_prefix(roi_name: str) -> str:
    """
    Normalise a ROI filename so LH and RH versions match.
    E.g. 'lh.hv4.bin.nii.gz' and 'rh.hv4.bin.nii.gz' → 'hv4.bin.nii.gz'
    """
    return re.sub(r"^(?:lh|rh)[._]", "", roi_name, flags=re.IGNORECASE)


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def avg_lh_rh_csv(
    lh_csv: Path,
    rh_csv: Path,
    out_csv: Path,
    group: Optional[str] = None,
) -> Path:
    """
    Merge and average two hemisphere CSV files.

    Returns the path of the written output CSV.
    """
    lh_rows = _read_csv(lh_csv)
    rh_rows = _read_csv(rh_csv)

    # Index by normalised ROI base-name
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
            for r in [lh_row, rh_row]:
                if r is not None:
                    try:
                        vals.append(float(r[col]))
                    except (KeyError, ValueError, TypeError):
                        pass
            row_out[col] = round(sum(vals) / len(vals), 6) if vals else ""

        # Also record individual hemisphere values for reference
        for prefix, src_row in [("lh_", lh_row), ("rh_", rh_row)]:
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
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute bilateral averages from LH and RH VOF coverage CSVs."
    )
    ap.add_argument("--lh",    required=True, type=Path,
                    help="Left-hemisphere CSV (from vof_map_coverage.py)")
    ap.add_argument("--rh",    required=True, type=Path,
                    help="Right-hemisphere CSV (from vof_map_coverage.py)")
    ap.add_argument("--out",   required=True, type=Path,
                    help="Output bilateral CSV")
    ap.add_argument("--group", default=None,
                    help="Label for the ROI group (e.g. ventral, dorsal, lo)")
    args = ap.parse_args()

    for p in [args.lh, args.rh]:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    avg_lh_rh_csv(args.lh, args.rh, args.out, args.group)


if __name__ == "__main__":
    main()

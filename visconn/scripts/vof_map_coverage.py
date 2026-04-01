#!/usr/bin/env python3
"""
vof_map_coverage.py — Map VOF streamline coverage over cortical ROI masks.

For each ROI mask provided, computes:
  - n_roi_vox        : total ROI voxels
  - n_covered_vox    : ROI voxels intersecting the VOF track-density image (TDI)
  - coverage_pct     : 100 * n_covered_vox / n_roi_vox
  - mean_tdi         : mean TDI value within the ROI

Usage:
    python3 vof_map_coverage.py \
        --tck        clean_vof.tck \
        --ref        t1.nii.gz \
        --rois       rh.hv4.bin.nii.gz rh.v2.ventral.bin.nii.gz ... \
        --out-csv    results/rh_vof_ventral_coverage.csv \
        [--out-tdi   images/rh_vof_ventral_tdi.nii.gz] \
        [--group     ventral] \
        [--hemisphere rh] \
        [--dwi-lib   visconn/libraries/DWIlib.sh]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np


def _tckmap(
    tck_path: Path,
    ref_path: Path,
    out_tdi: Path,
) -> Path:
    """
    Create a Track Density Image (TDI) from a .tck file using MRtrix tckmap.
    """
    out_tdi.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tckmap",
        str(tck_path),
        str(out_tdi),
        "-template", str(ref_path),
        "-quiet",
        "-force",
    ]
    print("[vof_map_coverage] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tdi


def coverage_stats(
    tdi_data: np.ndarray,
    roi_data: np.ndarray,
) -> dict:
    """
    Compute coverage statistics for one ROI given a TDI.

    Returns dict with keys: n_roi_vox, n_covered_vox, coverage_pct, mean_tdi.
    """
    roi_bin = (roi_data > 0)
    tdi_pos = (tdi_data > 0)

    n_roi = int(roi_bin.sum())
    if n_roi == 0:
        return {
            "n_roi_vox": 0,
            "n_covered_vox": 0,
            "coverage_pct": 0.0,
            "mean_tdi": 0.0,
        }

    # Handle shape mismatches (reference space vs ROI space)
    min_shape = tuple(min(a, b) for a, b in zip(roi_bin.shape, tdi_pos.shape))
    sl = tuple(slice(0, s) for s in min_shape)
    roi_sl  = roi_bin[sl]
    tdi_sl  = tdi_pos[sl]
    tdi_val = tdi_data[sl]

    n_covered = int((roi_sl & tdi_sl).sum())
    mean_tdi  = float(tdi_val[roi_sl].mean()) if roi_sl.any() else 0.0

    return {
        "n_roi_vox":    n_roi,
        "n_covered_vox": n_covered,
        "coverage_pct":  100.0 * n_covered / n_roi if n_roi > 0 else 0.0,
        "mean_tdi":      round(mean_tdi, 6),
    }


def vof_map_coverage(
    tck_path: Path,
    ref_path: Path,
    roi_paths: List[Path],
    out_csv: Path,
    out_tdi: Optional[Path] = None,
    group: str = "",
    hemisphere: str = "",
    dwi_lib: Optional[Path] = None,
) -> Path:
    """
    Compute VOF coverage for all ROI masks and write a CSV.

    Parameters
    ----------
    tck_path   : Cleaned VOF tractogram (.tck).
    ref_path   : Reference volume (used as TDI template).
    roi_paths  : List of binary ROI mask NIfTI files.
    out_csv    : Output CSV file path.
    out_tdi    : Optional path to save the TDI NIfTI. If None, uses a temp file.
    group      : Label for the ROI group (e.g. "ventral", "dorsal", "lo").
    hemisphere : Hemisphere label (e.g. "lh", "rh").
    dwi_lib    : Path to DWIlib.sh (accepted for API compatibility; not used here
                 since we call tckmap directly).

    Returns
    -------
    Path to the written CSV file.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build TDI
    if out_tdi is None:
        _tmp = tempfile.NamedTemporaryFile(suffix="_tdi.nii.gz", delete=False)
        tdi_path = Path(_tmp.name)
        _tmp.close()
        _cleanup_tdi = True
    else:
        tdi_path = out_tdi
        tdi_path.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_tdi = False

    _tckmap(tck_path, ref_path, tdi_path)

    tdi_img  = nib.load(str(tdi_path))
    tdi_data = np.asarray(tdi_img.dataobj)

    rows = []
    for roi_path in roi_paths:
        if not roi_path.exists():
            print(f"[vof_map_coverage] WARNING: ROI not found, skipping: {roi_path}")
            continue
        roi_img  = nib.load(str(roi_path))
        roi_data = np.asarray(roi_img.dataobj)
        stats    = coverage_stats(tdi_data, roi_data)
        rows.append({
            "hemisphere":    hemisphere,
            "group":         group,
            "roi":           roi_path.name,
            **stats,
        })
        print(
            f"[vof_map_coverage]  {roi_path.name:45s}  "
            f"coverage={stats['coverage_pct']:6.2f}%  "
            f"({stats['n_covered_vox']}/{stats['n_roi_vox']} vox)"
        )

    fieldnames = [
        "hemisphere", "group", "roi",
        "n_roi_vox", "n_covered_vox", "coverage_pct", "mean_tdi",
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if _cleanup_tdi:
        tdi_path.unlink(missing_ok=True)

    print(f"[vof_map_coverage] CSV written → {out_csv}  ({len(rows)} ROIs)")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map VOF streamline coverage over cortical ROI masks."
    )
    ap.add_argument("--tck",        required=True, type=Path,
                    help="Cleaned VOF tractogram (.tck)")
    ap.add_argument("--ref",        required=True, type=Path,
                    help="Reference volume for TDI template")
    ap.add_argument("--rois",       required=True, nargs="+", type=Path,
                    help="Binary ROI mask NIfTI files")
    ap.add_argument("--out-csv",    required=True, type=Path,
                    help="Output CSV file")
    ap.add_argument("--out-tdi",    type=Path, default=None,
                    help="Optional: save TDI NIfTI to this path")
    ap.add_argument("--group",      default="",
                    help="Label for the ROI group (e.g. ventral, dorsal, lo)")
    ap.add_argument("--hemisphere", default="",
                    help="Hemisphere label (e.g. lh, rh)")
    ap.add_argument("--dwi-lib",
                    default="visconn/libraries/DWIlib.sh",
                    help="Path to DWIlib.sh (accepted for compatibility)")
    args = ap.parse_args()

    if not args.tck.exists():
        raise FileNotFoundError(f"Tractogram not found: {args.tck}")
    if not args.ref.exists():
        raise FileNotFoundError(f"Reference volume not found: {args.ref}")

    vof_map_coverage(
        tck_path=args.tck,
        ref_path=args.ref,
        roi_paths=args.rois,
        out_csv=args.out_csv,
        out_tdi=args.out_tdi,
        group=args.group,
        hemisphere=args.hemisphere,
        dwi_lib=Path(args.dwi_lib) if args.dwi_lib else None,
    )


if __name__ == "__main__":
    main()

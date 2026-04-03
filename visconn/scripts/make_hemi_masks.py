#!/usr/bin/env python3
"""
make_hemi_masks.py — Generate left and right hemisphere binary masks
from a parcellation volume (aparc) and a label JSON file.

Usage:
    python3 make_hemi_masks.py --aparc aparc.nii.gz --label label.json \
        --out-dir /path/to/outdir

Outputs:
    <out-dir>/hemi_L.nii.gz
    <out-dir>/hemi_R.nii.gz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np


EXCLUDE_NAME_SUBSTRINGS = (
    "Ventricle",
    "ventricle",
    "CSF",
    "Vessel",
    "vessel",
    "Choroid-Plexus",
    "choroid",
    "Unknown",
    "unknown",
    "Lesion",
    "WM-hypointensities",
    "non-WM-hypointensities",
    "Optic-Chiasm",
    "Exterior",
)


def load_label_json(label_path: Path) -> Dict[int, str]:
    """Load label.json and return a dict mapping voxel_value -> name."""
    raw = json.loads(label_path.read_text())

    if isinstance(raw, list):
        out = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            if "voxel_value" in entry and "name" in entry:
                try:
                    out[int(entry["voxel_value"])] = str(entry["name"])
                except (ValueError, TypeError):
                    pass
        if out:
            return out

    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "name" in v:
                try:
                    out[int(k)] = str(v["name"])
                except (ValueError, TypeError):
                    pass
            elif isinstance(v, str):
                try:
                    out[int(k)] = v
                except (ValueError, TypeError):
                    pass
        if out:
            return out

    raise ValueError(
        f"Unrecognized label.json format in {label_path}. "
        "Expected a list of {voxel_value, name} objects or a mapping."
    )


def load_3d_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI image and collapse only a singleton trailing 4th dim."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, or 4D with singleton last axis, "
            f"but got shape {data.shape} from {path}"
        )

    return img, data


def should_exclude_label(name: str) -> bool:
    return any(s in name for s in EXCLUDE_NAME_SUBSTRINGS)


def collect_hemi_values(label_map: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build left/right label sets from FreeSurfer-style names and values.

    Includes:
      - names beginning with Left- / Right-
      - cortical parcel conventions:
          1000-1999 -> left
          2000-2999 -> right

    Excludes some obvious non-parenchymal / undesirable structures by name.
    """
    left_values = set()
    right_values = set()

    for vv, name in label_map.items():
        if should_exclude_label(name):
            continue

        if name.startswith("Left-"):
            left_values.add(vv)
        elif name.startswith("Right-"):
            right_values.add(vv)

        # FreeSurfer cortical annotation conventions
        if 1000 <= vv < 2000:
            left_values.add(vv)
        elif 2000 <= vv < 3000:
            right_values.add(vv)

    if not left_values:
        raise ValueError("No left-hemisphere labels found after filtering.")
    if not right_values:
        raise ValueError("No right-hemisphere labels found after filtering.")

    return np.array(sorted(left_values), dtype=np.int32), np.array(sorted(right_values), dtype=np.int32)


def make_hemi_masks(
    aparc_path: Path,
    label_path: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_json(label_path)
    left_values, right_values = collect_hemi_values(label_map)

    print(f"[make_hemi_masks] Left labels  ({len(left_values)}): {left_values.tolist()}")
    print(f"[make_hemi_masks] Right labels ({len(right_values)}): {right_values.tolist()}")

    img, data = load_3d_nifti(aparc_path)

    # FreeSurfer aparc values are labels, but may be stored as float-like arrays.
    # Round to nearest integer before matching.
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    else:
        data = data.astype(np.int32, copy=False)

    lh_mask = np.isin(data, left_values).astype(np.uint8)
    rh_mask = np.isin(data, right_values).astype(np.uint8)

    overlap = int(np.count_nonzero(lh_mask & rh_mask))
    if overlap != 0:
        raise ValueError(f"Left/right masks overlap in {overlap} voxels, which should not happen.")

    lh_path = out_dir / "hemi_L.nii.gz"
    rh_path = out_dir / "hemi_R.nii.gz"

    for out_path, mask in ((lh_path, lh_mask), (rh_path, rh_mask)):
        out_img = nib.Nifti1Image(mask, img.affine, img.header.copy())
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(out_path))
        print(f"[make_hemi_masks] Saved {out_path.name}  ({int(mask.sum())} voxels)")

    return lh_path, rh_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate hemisphere binary masks from aparc parcellation + label.json"
    )
    ap.add_argument("--aparc", required=True, type=Path,
                    help="Parcellation volume (aparc.nii.gz)")
    ap.add_argument("--label", required=True, type=Path,
                    help="Label JSON mapping voxel values to region names")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory")
    args = ap.parse_args()

    if not args.aparc.exists():
        raise FileNotFoundError(f"aparc not found: {args.aparc}")
    if not args.label.exists():
        raise FileNotFoundError(f"label.json not found: {args.label}")

    lh_path, rh_path = make_hemi_masks(args.aparc, args.label, args.out_dir)
    print(f"[make_hemi_masks] Done. LH: {lh_path}  RH: {rh_path}")


if __name__ == "__main__":
    main()

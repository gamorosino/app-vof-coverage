#!/usr/bin/env python3
"""
make_hemi_masks.py — Generate left and right hemisphere binary masks
from a parcellation volume (aparc) and a label JSON file.

Label JSON format expected (FreeSurfer-style):
    { "voxel_value": 1, "name": "Left-Cerebral-White-Matter", ... }
    or as an array:
    [ { "voxel_value": 1, "name": "Left-Cerebral-White-Matter" }, ... ]

Usage:
    python3 make_hemi_masks.py --aparc aparc.nii.gz --label label.json \
        --out-dir /path/to/outdir

Outputs:
    <out-dir>/hemi_L.nii.gz  — binary mask, 1 where aparc is a Left-* label
    <out-dir>/hemi_R.nii.gz  — binary mask, 1 where aparc is a Right-* label
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_label_json(label_path: Path) -> dict:
    """Load label.json and return a dict mapping voxel_value → name."""
    raw = json.loads(label_path.read_text())
    if isinstance(raw, list):
        return {entry["voxel_value"]: entry["name"] for entry in raw}
    if isinstance(raw, dict):
        # Could be { "value": { ... } } or direct mapping
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "name" in v:
                try:
                    result[int(k)] = v["name"]
                except (ValueError, TypeError):
                    pass
            elif isinstance(v, str):
                try:
                    result[int(k)] = v
                except (ValueError, TypeError):
                    pass
        if result:
            return result
    raise ValueError(
        f"Unrecognised label.json format in {label_path}. "
        "Expected a list of {{voxel_value, name}} objects or a mapping."
    )


def make_hemi_masks(
    aparc_path: Path,
    label_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Create hemi_L.nii.gz and hemi_R.nii.gz in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_json(label_path)

    left_values = {
        vv for vv, name in label_map.items() if name.startswith("Left-")
    }
    right_values = {
        vv for vv, name in label_map.items() if name.startswith("Right-")
    }

    if not left_values:
        raise ValueError("No labels starting with 'Left-' found in label.json")
    if not right_values:
        raise ValueError("No labels starting with 'Right-' found in label.json")

    print(f"[make_hemi_masks] Left labels  ({len(left_values)}): {sorted(left_values)}")
    print(f"[make_hemi_masks] Right labels ({len(right_values)}): {sorted(right_values)}")

    img = nib.load(str(aparc_path))
    data = np.asarray(img.dataobj)

    lh_mask = np.zeros(data.shape, dtype=np.uint8)
    rh_mask = np.zeros(data.shape, dtype=np.uint8)

    for vv in left_values:
        lh_mask[data == vv] = 1
    for vv in right_values:
        rh_mask[data == vv] = 1

    lh_path = out_dir / "hemi_L.nii.gz"
    rh_path = out_dir / "hemi_R.nii.gz"

    for out_path, mask in [(lh_path, lh_mask), (rh_path, rh_mask)]:
        out_img = nib.Nifti1Image(mask, img.affine, img.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(out_path))
        nvox = int(mask.sum())
        print(f"[make_hemi_masks] Saved {out_path.name}  ({nvox} voxels)")

    return lh_path, rh_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate hemisphere binary masks from aparc parcellation + label.json"
    )
    ap.add_argument("--aparc", required=True, type=Path,
                    help="Parcellation volume (aparc.nii.gz)")
    ap.add_argument("--label", required=True, type=Path,
                    help="Label JSON file mapping voxel values to region names")
    ap.add_argument("--out-dir", default=".", type=Path,
                    help="Output directory (default: current dir)")
    args = ap.parse_args()

    if not args.aparc.exists():
        raise FileNotFoundError(f"aparc not found: {args.aparc}")
    if not args.label.exists():
        raise FileNotFoundError(f"label.json not found: {args.label}")

    lh_path, rh_path = make_hemi_masks(args.aparc, args.label, args.out_dir)
    print(f"[make_hemi_masks] Done. LH: {lh_path}  RH: {rh_path}")


if __name__ == "__main__":
    main()

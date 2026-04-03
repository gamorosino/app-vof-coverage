#!/usr/bin/env python3
"""
make_roi_masks.py — Generate binary ROI masks for the VISCONN VOF pipeline.

Equivalent to benson14_v123_ventral_dorsal_nifti.sh, but parameterised so it
reads inputs from command-line arguments rather than hard-coded paths.

Benson14 varea label encoding used here (same as extract_template_tract_segment.py):
    V1=1, V2=2, V3=3, hV4=4, VO1=5, VO2=6, LO1=7, LO2=8,
    TO1=9, TO2=10, V3b=11, V3a=12

Polar-angle convention (Benson14):
    0°  = upper vertical meridian (UVM)
    90° = horizontal meridian (HM)
    180° = lower vertical meridian (LVM)

    Dorsal bank  (upper visual field) ← polar angle in [0°, 90°)
    Ventral bank (lower visual field) ← polar angle in (90°, 180°]

Usage:
    python3 make_roi_masks.py \
        --varea  benson14_varea.nii.gz \
        --angle  benson14_angle.nii.gz \
        --hemi-l hemi_L.nii.gz \
        --hemi-r hemi_R.nii.gz \
        --out-dir /path/to/masks/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# Benson14 area → voxel-value mapping
AREA_LABELS = [
    "V1", "V2", "V3", "hV4", "VO1", "VO2",
    "LO1", "LO2", "TO1", "TO2", "V3b", "V3a",
]
LABEL_TO_VAL = {lab: i + 1 for i, lab in enumerate(AREA_LABELS)}

# Polar-angle threshold that separates dorsal from ventral banks (degrees)
DORSOVENTRAL_THRESHOLD = 90.0

def _load_nifti_array(path: Path, name: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)

    # Common case: empty/singleton 4th dimension
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"{name} must be 3D after squeeze, but got shape {arr.shape} from {path}"
        )

    return img, arr

def _save_mask(data: np.ndarray, ref_img, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    nvox = int(data.sum())
    print(f"[make_roi_masks] {out_path.name:50s}  {nvox:6d} vox")
    return out_path


def make_roi_masks(
    varea_path: Path,
    angle_path: Path,
    hemi_l_path: Path,
    hemi_r_path: Path,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Generate all ROI masks needed by the VISCONN pipeline.

    Returns a dict mapping mask name (without .nii.gz) to its path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    varea_img, varea = _load_nifti_array(varea_path, "varea")
    angle_img, angle = _load_nifti_array(angle_path, "angle")
    _, lh = _load_nifti_array(hemi_l_path, "hemi_l")
    _, rh = _load_nifti_array(hemi_r_path, "hemi_r")
    
    lh = lh > 0
    rh = rh > 0

    masks: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Helper — build a name like rh.v2.ventral and save it
    # ------------------------------------------------------------------
    def _emit(hemi_tag: str, area_tag: str, binary: np.ndarray) -> None:
        name = f"{hemi_tag}.{area_tag}.bin"
        path = _save_mask(binary, varea_img, out_dir / f"{name}.nii.gz")
        masks[name] = path

    # ------------------------------------------------------------------
    # Dorsal / ventral polar-angle masks
    # ------------------------------------------------------------------
    dorsal_angle  = (angle >= 0) & (angle < DORSOVENTRAL_THRESHOLD)
    ventral_angle = (angle >= DORSOVENTRAL_THRESHOLD) & (angle <= 180)

    # Also handle images where angle is in [0, 360] by treating
    # values > 180 as mirrored (symmetric ventral/dorsal split):
    ventral_angle |= (angle > 180) & (angle <= 270)

    # ------------------------------------------------------------------
    # Per-hemisphere, per-area masks
    # ------------------------------------------------------------------
    for hemi_tag, hemi_mask in [("lh", lh), ("rh", rh)]:
        for area, val in LABEL_TO_VAL.items():
            area_vox = varea == val

            # Simple area masks (hV4, VO1, VO2, LO1, LO2, TO1, TO2, V3a, V3b, V1)
            simple_areas = {
                "hv4":  "hV4",
                "vo1":  "VO1",
                "vo2":  "VO2",
                "lo1":  "LO1",
                "lo2":  "LO2",
                "to1":  "TO1",
                "to2":  "TO2",
                "v3a":  "V3a",
                "v3b":  "V3b",
                "v1":   "V1",
            }
            area_lower = area.lower()
            if area_lower in simple_areas:
                mask = (area_vox & hemi_mask).astype(np.uint8)
                _emit(hemi_tag, area_lower, mask)

            # Ventral/dorsal split for V2 and V3
            if area in ("V2", "V3"):
                tag = area.lower()
                ventral_mask = (area_vox & ventral_angle & hemi_mask).astype(np.uint8)
                dorsal_mask  = (area_vox & dorsal_angle  & hemi_mask).astype(np.uint8)
                _emit(hemi_tag, f"{tag}.ventral", ventral_mask)
                _emit(hemi_tag, f"{tag}.dorsal",  dorsal_mask)

    return masks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate binary ROI masks for the VISCONN VOF pipeline."
    )
    ap.add_argument("--varea", required=True, type=Path,
                    help="Benson14 visual area map (varea.nii.gz)")
    ap.add_argument("--angle", required=True, type=Path,
                    help="Benson14 polar angle map (polarAngle.nii.gz)")
    ap.add_argument("--hemi-l", required=True, type=Path,
                    help="Left-hemisphere binary mask (hemi_L.nii.gz)")
    ap.add_argument("--hemi-r", required=True, type=Path,
                    help="Right-hemisphere binary mask (hemi_R.nii.gz)")
    ap.add_argument("--out-dir", default=".", type=Path,
                    help="Output directory for ROI masks")
    args = ap.parse_args()

    for p in [args.varea, args.angle, args.hemi_l, args.hemi_r]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    masks = make_roi_masks(
        args.varea, args.angle, args.hemi_l, args.hemi_r, args.out_dir
    )
    print(f"[make_roi_masks] Done. {len(masks)} masks written to {args.out_dir}")


if __name__ == "__main__":
    main()

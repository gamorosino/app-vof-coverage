#!/usr/bin/env python3
"""
make_vof_waypoints.py — Generate waypoint masks for the VOF (Vertical Occipital
Fasciculus) pipeline.

The VOF connects dorsal visual cortex (V3a/V3b/V2-dorsal/V3-dorsal) to ventral
visual cortex (hV4/V2-ventral/V3-ventral) in the lateral occipital lobe.

Two waypoint NIfTI masks are produced:
    wp_ventral.nii.gz  — union of ventral-side ROI masks intersected with WM
    wp_dorsal.nii.gz   — union of dorsal-side ROI masks intersected with WM

Usage:
    python3 make_vof_waypoints.py \
        --wm          wm_mask.nii.gz \
        --rois-dir    /path/to/roi/masks/ \
        --hemisphere  rh \
        --out-dir     /path/to/waypoints/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np


# Default sets of ROI names that form each waypoint (per-hemisphere prefix added)
VENTRAL_ROIS = ["hv4", "v2.ventral", "v3.ventral", "vo1", "vo2"]
DORSAL_ROIS  = ["v3a", "v3b", "v2.dorsal", "v3.dorsal"]



def _load_bin(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None

    arr = np.asarray(nib.load(str(path)).dataobj)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"ROI mask must be 3D after squeeze, but got shape {arr.shape} from {path}"
        )

    return (arr > 0).astype(np.uint8)


def _union(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    """Return element-wise OR of all arrays (broadcast to common shape)."""
    if not arrays:
        return None
    result = arrays[0].copy()
    for a in arrays[1:]:
        min_shape = tuple(min(x, y) for x, y in zip(result.shape, a.shape))
        sl = tuple(slice(0, s) for s in min_shape)
        result = (result[sl] | a[sl]).astype(np.uint8)
    return result


def _save(data: np.ndarray, ref_img, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    print(f"[make_vof_waypoints] Saved {out_path.name}  ({int(data.sum())} vox)")
    return out_path


def make_vof_waypoints(
    wm_mask_path: Optional[Path],
    rois_dir: Path,
    hemisphere: str,
    out_dir: Path,
    ventral_rois: Optional[List[str]] = None,
    dorsal_rois: Optional[List[str]] = None,
) -> tuple[Path, Path]:
    """
    Build wp_ventral and wp_dorsal waypoint masks for the given hemisphere.

    Parameters
    ----------
    wm_mask_path  : Optional binary WM mask. Accepted for backward compatibility
                    but not used to expand waypoint masks.
    rois_dir      : Directory that contains the per-hemisphere ROI .nii.gz files.
    hemisphere    : "lh" or "rh".
    out_dir       : Where to write the output waypoint masks.
    ventral_rois  : List of ROI tag strings for the ventral waypoint.
    dorsal_rois   : List of ROI tag strings for the dorsal waypoint.

    Returns
    -------
    (wp_ventral_path, wp_dorsal_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hemi = hemisphere.lower()

    if ventral_rois is None:
        ventral_rois = VENTRAL_ROIS
    if dorsal_rois is None:
        dorsal_rois = DORSAL_ROIS

    def _load_group(roi_tags: List[str]) -> tuple[List[np.ndarray], Optional[nib.Nifti1Image]]:
        loaded = []
        ref_img = None
        for tag in roi_tags:
            candidate = rois_dir / f"{hemi}.{tag}.bin.nii.gz"
            if not candidate.exists():
                print(f"[make_vof_waypoints]   (missing, skipped) {candidate.name}")
                continue

            img = nib.load(str(candidate))
            arr = np.asarray(img.dataobj)
            arr = np.squeeze(arr)

            if arr.ndim != 3:
                raise ValueError(
                    f"ROI mask must be 3D after squeeze, but got shape {arr.shape} from {candidate}"
                )

            arr = (arr > 0).astype(np.uint8)

            if ref_img is None:
                ref_img = img
            else:
                if arr.shape != loaded[0].shape:
                    raise ValueError(
                        f"ROI mask shape mismatch: {candidate.name} has shape {arr.shape}, "
                        f"expected {loaded[0].shape}"
                    )

            print(f"[make_vof_waypoints]   loaded {candidate.name}")
            loaded.append(arr)

        return loaded, ref_img

    ventral_arrays, ventral_ref = _load_group(ventral_rois)
    dorsal_arrays, dorsal_ref = _load_group(dorsal_rois)

    if not ventral_arrays:
        raise RuntimeError(
            f"No ventral ROI masks found in {rois_dir} for hemisphere '{hemi}'. "
            f"Expected any of: {[hemi + '.' + r + '.bin.nii.gz' for r in ventral_rois]}"
        )
    if not dorsal_arrays:
        raise RuntimeError(
            f"No dorsal ROI masks found in {rois_dir} for hemisphere '{hemi}'. "
            f"Expected any of: {[hemi + '.' + r + '.bin.nii.gz' for r in dorsal_rois]}"
        )

    if ventral_arrays[0].shape != dorsal_arrays[0].shape:
        raise ValueError(
            f"Ventral/dorsal ROI grids do not match: "
            f"{ventral_arrays[0].shape} vs {dorsal_arrays[0].shape}"
        )

    ref_img = ventral_ref if ventral_ref is not None else dorsal_ref
    if ref_img is None:
        raise RuntimeError("Could not determine a reference image from ROI masks.")

    ventral_union = _union(ventral_arrays)
    dorsal_union = _union(dorsal_arrays)

    # NOTE:
    # wm_mask_path is accepted for backward compatibility, but waypoint masks are
    # no longer expanded with WM because WM may be in a different image space.
    if wm_mask_path is not None:
        print(
            f"[make_vof_waypoints] WM mask provided ({wm_mask_path}) but ignored "
            f"for waypoint construction."
        )

    wp_ventral_data = (ventral_union > 0).astype(np.uint8)
    wp_dorsal_data = (dorsal_union > 0).astype(np.uint8)

    wp_ventral_path = out_dir / f"{hemi}_wp_ventral.nii.gz"
    wp_dorsal_path = out_dir / f"{hemi}_wp_dorsal.nii.gz"

    _save(wp_ventral_data, ref_img, wp_ventral_path)
    _save(wp_dorsal_data, ref_img, wp_dorsal_path)

    return wp_ventral_path, wp_dorsal_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ventral and dorsal VOF waypoint masks."
    )
    ap.add_argument("--wm", required=True, type=Path,
                    help="WM binary mask (wm_mask.nii.gz)")
    ap.add_argument("--rois-dir", required=True, type=Path,
                    help="Directory containing per-hemisphere ROI .nii.gz masks")
    ap.add_argument("--hemisphere", required=True, choices=["lh", "rh"],
                    help="Hemisphere to process")
    ap.add_argument("--out-dir", default=".", type=Path,
                    help="Output directory for waypoint masks")
    ap.add_argument("--ventral-rois", nargs="+", default=None,
                    help="Override default ventral ROI tag list")
    ap.add_argument("--dorsal-rois", nargs="+", default=None,
                    help="Override default dorsal ROI tag list")
    args = ap.parse_args()

    if not args.wm.exists():
        raise FileNotFoundError(f"WM mask not found: {args.wm}")
    if not args.rois_dir.is_dir():
        raise FileNotFoundError(f"ROI directory not found: {args.rois_dir}")

    wp_v, wp_d = make_vof_waypoints(
        wm_mask_path=args.wm,
        rois_dir=args.rois_dir,
        hemisphere=args.hemisphere,
        out_dir=args.out_dir,
        ventral_rois=args.ventral_rois,
        dorsal_rois=args.dorsal_rois,
    )
    print(f"[make_vof_waypoints] Done.")
    print(f"  Ventral waypoint: {wp_v}")
    print(f"  Dorsal waypoint:  {wp_d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
clean_vof_like_takemura.py — Clean a VOF tractogram following the approach
described in Takemura et al. (2016).

Cleaning steps applied:
  1. Remove streamlines whose length is below --min-length mm.
  2. Remove streamlines that extend beyond the allowed z-coordinate range
     [z1, z2] in voxel space of the reference volume.
  3. (Optional) Remove streamlines passing through the contralateral hemisphere
     using --exclude-mask.

Usage:
    python3 clean_vof_like_takemura.py \
        --in   raw_vof.tck \
        --ref  t1.nii.gz \
        --out  clean_vof.tck \
        --z1 0 --z2 11 \
        [--min-length 20] \
        [--exclude-mask hemi_contralateral.nii.gz]
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def _tcklen_filter(in_tck: Path, out_tck: Path, min_len: float) -> Path:
    """Keep only streamlines longer than min_len mm using tckedit."""
    cmd = [
        "tckedit", str(in_tck), str(out_tck),
        "-minlength", str(min_len),
        "-quiet",
    ]
    print("[clean_vof] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tck


def _z_filter(
    in_tck: Path,
    out_tck: Path,
    ref_img: nib.Nifti1Image,
    z1: float,
    z2: float,
) -> Path:
    """
    Remove streamlines whose endpoints fall OUTSIDE voxel z-range [z1, z2].

    This is done by building a NIfTI mask of the valid z-slice range and using
    tckedit -include with that mask.

    If z1 == z2 == 0 the filter is a no-op (disabled).
    """
    if z1 == 0 and z2 == 0:
        import shutil
        shutil.copy2(str(in_tck), str(out_tck))
        return out_tck

    # Build an inclusion mask: 1 where z-voxel index is within [z1, z2]
    data = np.asarray(ref_img.dataobj)
    shape = data.shape[:3]

    z_mask = np.zeros(shape, dtype=np.uint8)
    z_lo = max(0, int(z1))
    z_hi = min(shape[2] - 1, int(z2))
    z_mask[:, :, z_lo : z_hi + 1] = 1

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_mask_path = Path(tmp.name)

    mask_img = nib.Nifti1Image(z_mask, ref_img.affine, ref_img.header)
    mask_img.set_data_dtype(np.uint8)
    nib.save(mask_img, str(tmp_mask_path))

    cmd = [
        "tckedit", str(in_tck), str(out_tck),
        "-include", str(tmp_mask_path),
        "-quiet",
    ]
    print("[clean_vof] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_mask_path.unlink(missing_ok=True)

    return out_tck


def _exclude_filter(in_tck: Path, out_tck: Path, exclude_mask: Path) -> Path:
    """Exclude streamlines passing through a mask (e.g. contralateral hemisphere)."""
    cmd = [
        "tckedit", str(in_tck), str(out_tck),
        "-exclude", str(exclude_mask),
        "-quiet",
    ]
    print("[clean_vof] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tck


def _count(tck: Path) -> int:
    try:
        out = subprocess.check_output(
            ["tckinfo", str(tck), "-count"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split()[-1])
    except Exception:
        return -1


def clean_vof(
    in_tck: Path,
    ref_path: Path,
    out_tck: Path,
    z1: float = 0,
    z2: float = 0,
    min_length: float = 20.0,
    exclude_mask: Path | None = None,
) -> Path:
    """Run the full VOF cleaning pipeline."""
    out_tck.parent.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(str(ref_path))

    n_in = _count(in_tck)
    print(f"[clean_vof] Input streamlines: {n_in}")

    current = in_tck

    # Step 1: length filter
    if min_length > 0:
        tmp1 = out_tck.with_name(out_tck.stem + "_step1_len.tck")
        _tcklen_filter(current, tmp1, min_length)
        print(f"[clean_vof] After length filter (>{min_length} mm): {_count(tmp1)}")
        current = tmp1

    # Step 2: z-coordinate range filter
    tmp2 = out_tck.with_name(out_tck.stem + "_step2_z.tck")
    _z_filter(current, tmp2, ref_img, z1, z2)
    print(f"[clean_vof] After z-range filter [z1={z1}, z2={z2}]: {_count(tmp2)}")
    current = tmp2

    # Step 3: optional exclusion mask
    if exclude_mask is not None and exclude_mask.exists():
        tmp3 = out_tck.with_name(out_tck.stem + "_step3_excl.tck")
        _exclude_filter(current, tmp3, exclude_mask)
        print(f"[clean_vof] After exclusion mask filter: {_count(tmp3)}")
        current = tmp3

    # Move final result to requested output path
    import shutil
    shutil.move(str(current), str(out_tck))

    # Clean up intermediate files
    for tmp in [
        out_tck.with_name(out_tck.stem + "_step1_len.tck"),
        out_tck.with_name(out_tck.stem + "_step2_z.tck"),
        out_tck.with_name(out_tck.stem + "_step3_excl.tck"),
    ]:
        if tmp != out_tck and tmp.exists():
            tmp.unlink()

    n_out = _count(out_tck)
    print(f"[clean_vof] Output streamlines: {n_out}  → {out_tck}")
    return out_tck


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean a VOF tractogram (Takemura-style)."
    )
    ap.add_argument("--in",   dest="in_tck",  required=True, type=Path,
                    help="Input tractogram (.tck)")
    ap.add_argument("--ref",  required=True, type=Path,
                    help="Reference volume (e.g. t1.nii.gz) for voxel-space filters")
    ap.add_argument("--out",  required=True, type=Path,
                    help="Output cleaned tractogram (.tck)")
    ap.add_argument("--z1",   type=float, default=0,
                    help="Minimum voxel z-index for endpoint filter (default: 0 = disabled)")
    ap.add_argument("--z2",   type=float, default=0,
                    help="Maximum voxel z-index for endpoint filter (default: 0 = disabled)")
    ap.add_argument("--min-length", type=float, default=20.0,
                    help="Minimum streamline length in mm (default: 20)")
    ap.add_argument("--exclude-mask", type=Path, default=None,
                    help="Binary mask for streamlines to exclude (e.g. contralateral hemi)")
    args = ap.parse_args()

    if not args.in_tck.exists():
        raise FileNotFoundError(f"Input tractogram not found: {args.in_tck}")
    if not args.ref.exists():
        raise FileNotFoundError(f"Reference volume not found: {args.ref}")

    clean_vof(
        in_tck=args.in_tck,
        ref_path=args.ref,
        out_tck=args.out,
        z1=args.z1,
        z2=args.z2,
        min_length=args.min_length,
        exclude_mask=args.exclude_mask,
    )


if __name__ == "__main__":
    main()

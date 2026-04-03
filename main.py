#!/usr/bin/env python3
"""
main.py — VISCONN pipeline: VOF coverage mapping from a template tractogram.

Full pipeline:
  1.  Read config.json (Brainlife-style)
  2.  Generate hemisphere masks (hemi_L.nii.gz, hemi_R.nii.gz)
  3.  Generate ROI masks from Benson14 varea + polarAngle
  4.  Generate WM mask from tractogram (track density → binarise)
  5.  For each hemisphere (lh, rh):
      a. Generate VOF waypoint masks (wp_ventral, wp_dorsal)
      b. tckedit  → raw VOF per hemisphere
      c. clean_vof_like_takemura  → cleaned VOF
  6.  For each hemisphere × group (ventral, dorsal, lo):
      - vof_map_coverage  → coverage CSV + TDI NIfTI
  7.  avg_lh_rh_csv  → bilateral average CSVs per group

Outputs (Brainlife datatype dirs):
  images/   — TDI NIfTI images per hemisphere × group
  vof/      — cleaned VOF tractograms (.tck) per hemisphere
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS    = SCRIPT_DIR / "visconn" / "scripts"
LIBRARIES  = SCRIPT_DIR / "visconn" / "libraries"
DWI_LIB    = LIBRARIES / "DWIlib.sh"

MAKE_HEMI_MASKS   = SCRIPTS / "make_hemi_masks.py"
MAKE_ROI_MASKS    = SCRIPTS / "make_roi_masks.py"
MAKE_VWP          = SCRIPTS / "make_vof_waypoints.py"
CLEAN_VOF         = SCRIPTS / "clean_vof_like_takemura.py"
VOF_MAP_COVERAGE  = SCRIPTS / "vof_map_coverage.py"
AVG_CSV           = SCRIPTS / "avg_lh_rh_csv.py"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg(path: str | Path = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] config.json not found: {p}")
    return json.loads(p.read_text())


def _req(cfg: dict, *keys: str) -> Path:
    for k in keys:
        v = cfg.get(k)
        if v and v not in ("", "null"):
            p = Path(v)
            if not p.exists():
                sys.exit(f"[ERROR] Config key '{k}' → file not found: {p}")
            return p
    sys.exit(f"[ERROR] Missing required config key (tried: {keys})")


def _req_str(cfg: dict, *keys: str) -> str:
    for k in keys:
        v = cfg.get(k)
        if v and v not in ("", "null"):
            return str(v)
    sys.exit(f"[ERROR] Missing required config key (tried: {keys})")


def _opt(cfg: dict, *keys: str, default=None):
    for k in keys:
        v = cfg.get(k)
        if v and v not in ("", "null"):
            return v
    return default


# ---------------------------------------------------------------------------
# track_getMask: create a binary WM mask from a tractogram
# ---------------------------------------------------------------------------

def track_getMask(tractogram: Path, out_mask: Path, reference: Path) -> Path:
    """
    Generate a binary WM mask by computing the track-density image (TDI)
    of the input tractogram and binarising it.
    """
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    # Use DWIlib.sh if available, otherwise call tckmap directly
    if DWI_LIB.exists():
        cmd = [
            "bash", "-c",
            f'source "{DWI_LIB}" && '
            f'track_getMask "{tractogram}" "{out_mask}" "{reference}"',
        ]
        print("[track_getMask] Using DWIlib.sh")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and out_mask.exists():
            return out_mask
        print("[track_getMask] DWIlib.sh failed; falling back to tckmap+fslmaths")

    # Fallback: tckmap → density image → binarise
    tdi_tmp = out_mask.with_name(out_mask.stem + "_tdi.nii.gz")
    subprocess.run(
        ["tckmap", str(tractogram), str(tdi_tmp),
         "-template", str(reference), "-quiet", "-force"],
        check=True,
    )

    # Binarise via fslmaths if available, else via nibabel
    result = subprocess.run(
        ["fslmaths", str(tdi_tmp), "-bin", str(out_mask)],
        check=False,
    )
    if result.returncode != 0:
        img  = nib.load(str(tdi_tmp))
        data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
        out  = nib.Nifti1Image(data, img.affine, img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(out_mask))

    tdi_tmp.unlink(missing_ok=True)
    print(f"[track_getMask] WM mask → {out_mask}")
    return out_mask


# ---------------------------------------------------------------------------
# tckedit helper
# ---------------------------------------------------------------------------

def run_tckedit(
    tractogram: Path,
    out_tck: Path,
    include1: Path,
    include2: Path,
    exclude: Path,
    ends_only: bool = True,
) -> Path:
    """
    Extract streamlines passing through both include1 and include2 and NOT
    through the exclusion mask.
    """
    out_tck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tckedit", str(tractogram), str(out_tck),
        "-include", str(include1),
        "-include", str(include2),
        "-exclude", str(exclude),
        "-force"
    ]
    if ends_only:
        cmd.append("-ends_only")
    cmd.append("-quiet")
    print("[tckedit] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tck


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run_script(script: Path, *args):
    cmd = [sys.executable, str(script)] + [str(a) for a in args]
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# ROI group definitions
# ---------------------------------------------------------------------------

GROUPS = {
    "ventral": ["hv4", "v2.ventral", "v3.ventral", "vo1", "vo2"],
    "dorsal":  ["v3a", "v3b", "v2.dorsal", "v3.dorsal"],
    "lo":      ["lo1", "lo2"],
}


def _roi_paths_for_group(roi_dir: Path, hemi: str, group: str) -> list[Path]:
    tags = GROUPS.get(group, [])
    paths = []
    for tag in tags:
        p = roi_dir / f"{hemi}.{tag}.bin.nii.gz"
        if p.exists():
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="VISCONN VOF coverage pipeline (Brainlife app)."
    )
    ap.add_argument("--config", default="config.json",
                    help="Path to Brainlife config.json (default: config.json)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------
    tractogram   = _req(cfg, "tractogram", "track")
    t1           = _req(cfg, "t1")
    varea        = _req(cfg, "varea")
    polar_angle  = _req(cfg, "polarAngle", "polar_angle")
    aparc        = _req(cfg, "aparc")
    label_json   = _req(cfg, "label")

    # Optional but used if present
    r2           = _opt(cfg, "r2")
    eccentricity = _opt(cfg, "eccentricity")
    rf_width     = _opt(cfg, "rfWidth", "rf_width")
    use_wm_mask = _opt(cfg, "use_wm_mask", default=True)
    use_wm_mask = bool(use_wm_mask)
    # ------------------------------------------------------------------
    # Output directories (Brainlife-style)
    # ------------------------------------------------------------------
    images_dir = Path("images")
    vof_dir    = Path("vof")
    work_dir   = Path("work")

    for d in [images_dir, vof_dir, work_dir]:
        d.mkdir(parents=True, exist_ok=True)

    masks_dir   = work_dir / "masks"
    roi_dir     = work_dir / "roi_masks"
    wp_dir      = work_dir / "waypoints"
    raw_vof_dir = work_dir / "raw_vof"
    csv_dir     = work_dir / "csv"

    for d in [masks_dir, roi_dir, wp_dir, raw_vof_dir, csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Hemisphere masks
    # ------------------------------------------------------------------
    print("\n[VISCONN] === Step 1: Hemisphere masks ===")
    hemi_l = masks_dir / "hemi_L.nii.gz"
    hemi_r = masks_dir / "hemi_R.nii.gz"
    if not (hemi_l.exists() and hemi_r.exists()):
        _run_script(
            MAKE_HEMI_MASKS,
            "--aparc", aparc,
            "--label", label_json,
            "--out-dir", masks_dir,
        )
    else:
        print("[VISCONN] Hemisphere masks already exist, skipping.")

    # ------------------------------------------------------------------
    # Step 2 — ROI masks from Benson14 atlas
    # ------------------------------------------------------------------
    print("\n[VISCONN] === Step 2: ROI masks (Benson14 ventral/dorsal) ===")
    _run_script(
        MAKE_ROI_MASKS,
        "--varea",  varea,
        "--angle",  polar_angle,
        "--hemi-l", hemi_l,
        "--hemi-r", hemi_r,
        "--out-dir", roi_dir,
    )

    # ------------------------------------------------------------------
    # Step 3 — WM mask from tractogram
    # ------------------------------------------------------------------
    if use_wm_mask:
        print("\n[VISCONN] === Step 3: WM mask (track_getMask) ===")
        wm_mask = masks_dir / "wm_mask.nii.gz"
        if not wm_mask.exists():
            track_getMask(tractogram, wm_mask, t1)
        else:
            print("[VISCONN] WM mask already exists, skipping.")
    else:
        wm_mask = None
        print("\n[VISCONN] === Step 3: WM mask skipped (disabled) ===")

    # ------------------------------------------------------------------
    # Steps 4–6: Per-hemisphere VOF extraction and coverage mapping
    # ------------------------------------------------------------------
    bilateral_csvs: dict[str, dict[str, Path]] = {}

    for hemi in ("lh", "rh"):
        contralateral_hemi_mask = hemi_r if hemi == "lh" else hemi_l

        print(f"\n[VISCONN] === {hemi.upper()}: Waypoints ===")
        args_vwp = [
            "--rois-dir",   roi_dir,
            "--hemisphere", hemi,
            "--out-dir",    wp_dir,
        ]
        
        if use_wm_mask:
            print(f"[VISCONN] Using WM mask for waypoints: {wm_mask}")
            args_vwp = ["--wm", wm_mask] + args_vwp
        else:
            print("[VISCONN] WM mask disabled for waypoint construction")
        
        _run_script(MAKE_VWP, *args_vwp)
        wp_ventral = wp_dir / f"{hemi}_wp_ventral.nii.gz"
        wp_dorsal  = wp_dir / f"{hemi}_wp_dorsal.nii.gz"

        # ------------------------------------------------------------------
        # tckedit — extract raw VOF
        # ------------------------------------------------------------------
        print(f"\n[VISCONN] === {hemi.upper()}: tckedit ===")
        raw_vof = raw_vof_dir / f"{hemi}_vof_raw.tck"
        run_tckedit(
            tractogram   = tractogram,
            out_tck      = raw_vof,
            include1     = wp_ventral,
            include2     = wp_dorsal,
            exclude      = contralateral_hemi_mask,
            ends_only    = True,
        )

        # ------------------------------------------------------------------
        # clean_vof_like_takemura
        # ------------------------------------------------------------------
        print(f"\n[VISCONN] === {hemi.upper()}: Cleaning VOF (Takemura-style) ===")
        clean_tck = vof_dir / f"{hemi}_vof.tck"
        _run_script(
            CLEAN_VOF,
            "--in",  raw_vof,
            "--ref", t1,
            "--out", clean_tck,
            "--z1", "0",
            "--z2", "11",
            "--min-length", "20",
        )

        # ------------------------------------------------------------------
        # vof_map_coverage — per group
        # ------------------------------------------------------------------
        for group in ("ventral", "dorsal", "lo"):
            roi_list = _roi_paths_for_group(roi_dir, hemi, group)
            if not roi_list:
                print(f"[VISCONN] No ROIs for {hemi}/{group}, skipping.")
                continue

            print(f"\n[VISCONN] === {hemi.upper()}: Coverage mapping — {group} ===")
            out_csv = csv_dir / f"{hemi}_vof_{group}_coverage.csv"
            out_tdi = images_dir / f"{hemi}_vof_{group}_tdi.nii.gz"

            args_coverage = [
                "--tck",        clean_tck,
                "--ref",        t1,
                "--out-csv",    out_csv,
                "--out-tdi",    out_tdi,
                "--group",      group,
                "--hemisphere", hemi,
                "--dwi-lib",    DWI_LIB,
                "--rois",
            ] + roi_list
            _run_script(VOF_MAP_COVERAGE, *args_coverage)

            # Record CSV path for bilateral averaging
            bilateral_csvs.setdefault(group, {})[hemi] = out_csv

    # ------------------------------------------------------------------
    # Step 7 — Bilateral averages
    # ------------------------------------------------------------------
    print("\n[VISCONN] === Step 7: Bilateral averages ===")
    for group, hemi_map in bilateral_csvs.items():
        lh_csv = hemi_map.get("lh")
        rh_csv = hemi_map.get("rh")
        if lh_csv and rh_csv and lh_csv.exists() and rh_csv.exists():
            out_bilateral = csv_dir / f"bilateral_vof_{group}_coverage.csv"
            _run_script(
                AVG_CSV,
                "--lh",    lh_csv,
                "--rh",    rh_csv,
                "--out",   out_bilateral,
                "--group", group,
            )
            # Also copy to images dir for Brainlife
            # Copy CSV to images dir
            shutil.copy2(str(out_bilateral), str(images_dir / out_bilateral.name))
            
            # Copy associated figure(s) if they exist
            fig_prefix = out_bilateral.with_suffix("")
            if fig_prefix.suffix == ".nii":
                fig_prefix = fig_prefix.with_suffix("")
            
            for ext in (".png", ".pdf", ".svg"):
                fig_file = Path(str(fig_prefix) + ext)
                if fig_file.exists():
                    shutil.copy2(str(fig_file), str(images_dir / fig_file.name))
        else:
            print(f"[VISCONN] Skipping bilateral average for '{group}' "
                  "(one or both hemispheres missing).")

    print("\n[VISCONN] ✓ Pipeline complete.")
    print(f"  Tractograms : {vof_dir}/")
    print(f"  Images      : {images_dir}/")
    print(f"  CSVs        : {csv_dir}/")


if __name__ == "__main__":
    main()

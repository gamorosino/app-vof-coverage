#!/usr/bin/env python3
"""Shared utility functions for the VISCONN pipeline."""

from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Optional, Union
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt



def nature_style_plot(
    ax,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xticks=None,
    yticks=None,
    n_xticks=None,
    n_yticks=3,
    spine_width=2,
    tick_length=6,
    tick_width=2,
    fontsize=16,
    x_decimals=0,
    y_decimals=3,
    add_origin_padding=True,
    pad_fraction=0.02,
    format_xticklabels=True,
    format_yticklabels=True
):
    """
    Apply Nature-style formatting to matplotlib axes.

    Priority:
    1. explicit xticks / yticks
    2. n_xticks / n_yticks
    3. keep existing ticks
    """

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Style spines
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)

    # Tick style
    ax.tick_params(
        axis="both",
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=fontsize
    )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Set limits
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Optional padding
    if add_origin_padding:
        xmin_current, xmax_current = ax.get_xlim()
        ymin_current, ymax_current = ax.get_ylim()

        xpad = pad_fraction * (xmax_current - xmin_current)
        ypad = pad_fraction * (ymax_current - ymin_current)

        ax.set_xlim(xmin_current - xpad, xmax_current)
        ax.set_ylim(ymin_current - ypad, ymax_current)

    # ----- X ticks -----
    if xticks is not None:
        ax.set_xticks(xticks)
        if format_xticklabels:
            ax.set_xticklabels([f"{t:.{x_decimals}f}" for t in xticks])
    elif n_xticks is not None:
        xmin_current, xmax_current = ax.get_xlim()
        if n_xticks == 3:
            ticks = [xmin_current, (xmin_current + xmax_current) / 2, xmax_current]
        elif n_xticks == 2:
            ticks = [xmin_current, xmax_current]
        else:
            ticks = np.linspace(xmin_current, xmax_current, n_xticks)

        ax.set_xticks(ticks)
        if format_xticklabels:
            ax.set_xticklabels([f"{t:.{x_decimals}f}" for t in ticks])

    # ----- Y ticks -----
    if yticks is not None:
        ax.set_yticks(yticks)
        if format_yticklabels:
            ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in yticks])
    elif n_yticks is not None:
        ymin_current, ymax_current = ax.get_ylim()
        if n_yticks == 3:
            ticks = [ymin_current, (ymin_current + ymax_current) / 2, ymax_current]
        elif n_yticks == 2:
            ticks = [ymin_current, ymax_current]
        else:
            ticks = np.linspace(ymin_current, ymax_current, n_yticks)

        ax.set_yticks(ticks)
        if format_yticklabels:
            ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in ticks])

    # Trim spines to final ticks
    xticks_final = ax.get_xticks()
    yticks_final = ax.get_yticks()

    if len(xticks_final) > 0:
        ax.spines["bottom"].set_bounds(xticks_final[0], xticks_final[-1])

    if len(yticks_final) > 0:
        ax.spines["left"].set_bounds(yticks_final[0], yticks_final[-1])

    return ax

def save_figure(path, dpi=300):
    """Save plot in PNG and vector formats."""
    plt.savefig(path.with_suffix(".png"), dpi=dpi)
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".svg"))
# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path] = "config.json") -> dict:
    """Load and return the Brainlife config.json."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"config.json not found: {p}")
    return json.loads(p.read_text())


def cfg_get(cfg: dict, key: str, default=None):
    """Return cfg[key] or default."""
    return cfg.get(key, default)


def cfg_require_path(cfg: dict, *keys: str) -> Path:
    """Return the first matching config key as a resolved Path; raise if missing/absent."""
    for k in keys:
        v = cfg.get(k)
        if v and v not in ("null", ""):
            p = Path(v)
            if not p.exists():
                raise FileNotFoundError(
                    f"Config key '{k}' points to non-existent file: {p}"
                )
            return p
    raise ValueError(f"Missing required config key (tried: {keys})")


def cfg_require_str(cfg: dict, *keys: str) -> str:
    """Return the first matching config key as a string; raise if missing."""
    for k in keys:
        v = cfg.get(k)
        if v and v not in ("null", ""):
            return str(v)
    raise ValueError(f"Missing required config key (tried: {keys})")


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Union[str, Path]):
    """Load a NIfTI file and return (img, data, affine)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data, img.affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: Union[str, Path],
    ref_img=None,
    dtype=np.uint8,
):
    """Save a numpy array as a NIfTI file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ref_img is not None:
        img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    else:
        img = nib.Nifti1Image(data.astype(dtype), affine)
    img.set_data_dtype(dtype)
    nib.save(img, str(out_path))
    return out_path


def binarize(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return a uint8 binary array (1 where data > threshold, else 0)."""
    return (data > threshold).astype(np.uint8)


def union_masks(*paths) -> Optional[np.ndarray]:
    """Return the union (OR) of multiple binary mask arrays loaded from paths."""
    result = None
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        _, data, _ = load_nifti(p)
        m = binarize(data)
        result = m if result is None else (result | m).astype(np.uint8)
    return result


def intersect_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Return the intersection (AND) of two binary mask arrays."""
    min_shape = tuple(min(a, b) for a, b in zip(mask_a.shape, mask_b.shape))
    sl = tuple(slice(0, s) for s in min_shape)
    return (mask_a[sl] & mask_b[sl]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run(cmd, check=True, capture_output=False, **kwargs):
    """Run a subprocess command with logging."""
    print("[CMD] " + " ".join(str(c) for c in cmd))
    return subprocess.run(
        [str(c) for c in cmd],
        check=check,
        capture_output=capture_output,
        **kwargs,
    )


def count_streamlines(tck_path: Union[str, Path]) -> int:
    """Return the number of streamlines in a .tck file using tckinfo."""
    tck_path = Path(tck_path)
    if not tck_path.exists() or tck_path.stat().st_size == 0:
        return 0
    try:
        out = subprocess.check_output(
            ["tckinfo", str(tck_path), "-count"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split()[-1])
    except Exception:
        return 0

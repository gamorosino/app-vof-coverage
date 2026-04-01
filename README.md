# Vertical Occipital Fasciculus (VOF) Coverage Quantification

A [Brainlife](https://brainlife.io) app that extracts the **Vertical Occipital Fasciculus (VOF)** (Takemura et al. 2016) from a tractogram and computes its cortical coverage across retinotopic visual areas.

---

## Pipeline overview

```
config.json
   │
   ├─ aparc.nii.gz + label.json
   │       └─► make_hemi_masks.py ──► hemi_L.nii.gz, hemi_R.nii.gz
   │
   ├─ varea.nii.gz + polarAngle.nii.gz + hemi masks
   │       └─► make_roi_masks.py ──► rh.hv4.bin.nii.gz, rh.v2.ventral.bin.nii.gz, ...
   │
   ├─ tractogram.tck + t1.nii.gz
   │       └─► track_getMask (DWIlib) ──► wm_mask.nii.gz
   │
   └─ For each hemisphere (lh, rh):
           │
           ├─► make_vof_waypoints.py ──► {hemi}_wp_ventral.nii.gz, {hemi}_wp_dorsal.nii.gz
           │
           ├─► tckedit -include wp_ventral -include wp_dorsal
           │           -exclude contralateral_hemi ──► {hemi}_vof_raw.tck
           │
           ├─► clean_vof_like_takemura.py ──► vof/{hemi}_vof.tck
           │
           └─ For each group (ventral, dorsal, lo):
                   ├─► vof_map_coverage.py ──► images/{hemi}_vof_{group}_tdi.nii.gz
                   │                           work/csv/{hemi}_vof_{group}_coverage.csv
                   └─► avg_lh_rh_csv.py ──► work/csv/bilateral_vof_{group}_coverage.csv
```
---
## Author

Gabriele amorosino (g.amorosino@gmail.com)

---

## Required inputs (config.json)

| Key            | Description                                               | Example                          |
|----------------|-----------------------------------------------------------|----------------------------------|
| `tractogram`   | Template tractogram to process (`.tck`)                   | `../../track.tck`                |
| `t1`           | T1-weighted reference volume (MNI space, `.nii.gz`)       | `../../t1.nii.gz`                |
| `varea`        | Visual areas map (`.nii.gz`)                      | `../../prf/lh.benson14_varea.nii.gz` |
| `polarAngle`   | Polar angle map (`.nii.gz`)                      | `../../prf/lh.benson14_angle.nii.gz` |
| `aparc`        | FreeSurfer parcellation volume (`.nii.gz`)                | `../../parc/parc.nii.gz`         |
| `label`        | Label JSON mapping voxel values to region names           | `../../parc/label.json`          |

### Optional inputs

| Key            | Description                                               |
|----------------|-----------------------------------------------------------|
| `eccentricity` | Eccentricity map (`.nii.gz`)                     |
| `rfWidth`      | RF-width map (`.nii.gz`)                         |
| `r2`           | Variance-explained map (`.nii.gz`)               |
| `output`       | FreeSurfer output directory                               |
| `surfaces`     | Surface files directory                                   |
| `prf_surfaces` | PRF surface files directory                               |
| `key`          | Parcellation key file (`.txt`)                            |

### label.json format

The label JSON maps voxel integer values to FreeSurfer-style region names. Region names starting with `"Left-"` are used for the left hemisphere mask; `"Right-"` for the right.

```json
[
  { "voxel_value": 1, "name": "Left-Cerebral-White-Matter" },
  { "voxel_value": 2, "name": "Left-Cerebral-Cortex" },
  { "voxel_value": 41, "name": "Right-Cerebral-White-Matter" },
  { "voxel_value": 42, "name": "Right-Cerebral-Cortex" }
]
```

---

## Outputs

### Brainlife output datatypes

| Directory  | Contents                                                    |
|------------|-------------------------------------------------------------|
| `vof/`     | Cleaned VOF tractogram per hemisphere: `lh_vof.tck`, `rh_vof.tck` |
| `images/`  | Track-density images (TDI) per hemisphere × group; bilateral coverage CSVs |

### Coverage CSV columns

| Column          | Description                                        |
|-----------------|----------------------------------------------------|
| `hemisphere`    | `lh`, `rh`, or `bilateral`                         |
| `group`         | `ventral`, `dorsal`, or `lo`                       |
| `roi`           | ROI mask filename                                  |
| `n_roi_vox`     | Total ROI voxels                                   |
| `n_covered_vox` | ROI voxels intersecting the VOF TDI                |
| `coverage_pct`  | Percentage coverage: `100 × n_covered / n_roi`     |
| `mean_tdi`      | Mean TDI value within the ROI                      |

### ROI groups

| Group     | ROIs                                                        |
|-----------|-------------------------------------------------------------|
| `ventral` | hV4, V2 ventral, V3 ventral, VO1, VO2                       |
| `dorsal`  | V3a, V3b, V2 dorsal, V3 dorsal                              |
| `lo`      | LO1, LO2                                                   |

Dorsal vs ventral split of V2/V3 uses polar angle < 90° (dorsal) vs > 90° (ventral) following the Benson14 atlas convention.

---

## Running the app

### On Brainlife

The app is launched automatically by the Brainlife platform. The `main` script handles container invocation.

### Running locally

**Prerequisites**: Singularity, `jq`

```bash
# 1. Place config.json in the repo root (see config.json template)
# 2. Run
./main
```

### Without Singularity (development)

Make sure `micromamba`/`conda` environment `tract_align` is active with:
- Python packages: `nibabel`, `numpy`
- MRtrix3 (`tckedit`, `tckmap`, `tckinfo`)

```bash
python3 main.py --config config.json
```

### Container

```
docker://gamorosino/tract_align:v1.0
```

Override with:
```bash
BL_CONTAINER_IMAGE=docker://my-custom-image:tag ./main
```

---

## Repository structure

```
app-vof-coverage/
├── main              ← Brainlife bash entrypoint (Singularity launcher)
├── main.py           ← Full pipeline orchestrator
├── config.json       ← Example / template runtime config
├── README.md
└── visconn/
    ├── scripts/
    │   ├── utils.py                      ← Shared utilities
    │   ├── make_hemi_masks.py            ← aparc + label.json → hemi_L/R masks
    │   ├── make_roi_masks.py             ← varea + polarAngle → ROI masks
    │   ├── make_vof_waypoints.py         ← WM mask + ROIs → VOF waypoints
    │   ├── clean_vof_like_takemura.py    ← Tractogram cleaning (Takemura-style)
    │   ├── vof_map_coverage.py           ← VOF TDI → coverage CSV
    │   └── avg_lh_rh_csv.py             ← Bilateral average CSV
    └── libraries/
        ├── DWIlib.sh                     ← DWI/tractography helpers (track_getMask)
        ├── IMAGINGlib.sh                 ← Neuroimaging shell utilities
        ├── STRlib.sh                     ← String manipulation utilities
        ├── FILElib.sh                    ← File management utilities
        └── ARRAYlib.sh                   ← Array utilities
```

---

## References

- Takemura, H., Rokem, A., Winawer, J., Yeatman, J. D., Wandell, B. A., & Pestilli, F. (2016). A major human white matter pathway between dorsal and ventral visual cortex. Cerebral cortex, 26(5), 2205-2214.
- Tournier, J. D., Smith, R., Raffelt, D., Tabbara, R., Dhollander, T., Pietsch, M., ... & Connelly, A. (2019). MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. Neuroimage, 202, 116137.
- Hayashi, S., Caron, B. A., Heinsfeld, A. S., Vinci-Booher, S., McPherson, B., Bullock, D. N., ... & Pestilli, F. (2024). brainlife. io: A decentralized and open-source cloud platform to support neuroscience research. Nature methods, 21(5), 809-813.

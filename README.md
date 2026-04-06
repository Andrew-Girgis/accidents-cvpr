# accidents-cvpr

Local research workspace for the Kaggle **CVPR 2026 ACCIDENT** competition. This repo mixes dataset exploration, notebook-driven baselines, and lightweight reusable code for accident timing, localization, and collision-type prediction from CCTV video.

## Competition Summary

For each test video, the submission must predict:

1. `accident_time` in seconds
2. `center_x`, `center_y` in normalized frame coordinates
3. `type` in `{head-on, rear-end, sideswipe, t-bone, single}`

The leaderboard score is the harmonic mean of:

- temporal similarity `T`
- spatial similarity `S`
- classification accuracy `C`

This is a **zero-shot sim-to-real** problem: training labels come from synthetic CARLA videos, while the test set is real CCTV footage.

See [kaggle_comp.md](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/kaggle_comp.md) for the full competition spec and metric formula.

## Current Repository Layout

The repo is now organized so durable project files stay near the root and notebook experiments are self-contained.

```text
accidents-cvpr/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── NOTES.md
├── kaggle_comp.md
├── pyproject.toml
├── uv.lock
├── main.py
├── crash_signals.py
├── test_metadata.csv
├── notebooks/
│   ├── accident-detection-yolo26/
│   │   ├── accident_detection_yolo26.ipynb
│   │   └── kernel-metadata.json
│   ├── accident-detection-sam31/
│   │   ├── accident_detection_sam31.ipynb
│   │   └── kernel-metadata.json
│   ├── accident-r2plus1d-baseline/
│   │   ├── accident_r2plus1d_baseline.ipynb
│   │   └── kernel-metadata.json
│   └── r3d-accident-detection/
│       ├── r3d_accident_detection.ipynb
│       └── kernel-metadata.json
├── src/
│   └── accidents_cvpr/
│       ├── __init__.py
│       └── crash_signals.py
├── outputs/
├── checkpoints/
├── weights/
├── submissions/
├── archive/
├── sim_dataset/
└── videos/
```

## Notebook Convention

Each notebook lives in its own folder under `notebooks/` and includes:

- the `.ipynb`
- a colocated `kernel-metadata.json`

This keeps each experiment bundle self-contained for Kaggle upload/export and avoids loose notebook metadata files at the repo root.

## Data Layout

### Synthetic training data

The synthetic dataset lives under `sim_dataset/`:

```text
sim_dataset/
├── labels.csv
├── annotation_classes.yaml
├── videos/
└── video_annotations/
```

Important files:

- [sim_dataset/labels.csv](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/sim_dataset/labels.csv): training labels and impact metadata for 2,211 synthetic clips
- [sim_dataset/annotation_classes.yaml](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/sim_dataset/annotation_classes.yaml): object class IDs used in frame annotations

### Real test data

- `videos/`: real CCTV test clips
- [test_metadata.csv](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/test_metadata.csv): region, scene layout, weather, day/night, and quality metadata for the test set

### Local-only large assets

These are intentionally kept local and ignored by git:

- `sim_dataset/`
- `videos/`
- `weights/`
- `checkpoints/`
- `outputs/`
- `submissions/`
- `archive/`

## Experiment Inventory

### YOLO26 notebook

Path: [notebooks/accident-detection-yolo26/accident_detection_yolo26.ipynb](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/notebooks/accident-detection-yolo26/accident_detection_yolo26.ipynb)

Role:

- current submission-oriented notebook
- frame sampling and YOLO-based detection workflow
- writes submission artifacts and diagnostic outputs

Related local assets:

- [notebooks/accident-detection-yolo26/kernel-metadata.json](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/notebooks/accident-detection-yolo26/kernel-metadata.json)
- `weights/yolo26n-seg.pt`
- `outputs/` diagnostics
- `submissions/submission.csv`

### SAM 3.1 notebook

Path: [notebooks/accident-detection-sam31/accident_detection_sam31.ipynb](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/notebooks/accident-detection-sam31/accident_detection_sam31.ipynb)

Role:

- research notebook for detector-driven physics signals
- integrates the crash-signal logic from `crash_signals.py`
- useful for qualitative analysis and signal debugging

### R2Plus1D baseline

Path: [notebooks/accident-r2plus1d-baseline/accident_r2plus1d_baseline.ipynb](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/notebooks/accident-r2plus1d-baseline/accident_r2plus1d_baseline.ipynb)

Role:

- baseline temporal modeling notebook
- competition-structure-aware notebook for data inspection, training, and submission formatting

### R3D notebook

Path: [notebooks/r3d-accident-detection/r3d_accident_detection.ipynb](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/notebooks/r3d-accident-detection/r3d_accident_detection.ipynb)

Role:

- alternative temporal video baseline
- includes checkpoint-driven inference and submission generation logic

## Reusable Code

### `src/accidents_cvpr/crash_signals.py`

Path: [src/accidents_cvpr/crash_signals.py](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/src/accidents_cvpr/crash_signals.py)

This module contains the reusable accident-signal logic:

- track-based velocity drop
- optical-flow anomaly detection
- trajectory intersection and heading classification
- signal fusion utilities
- a lightweight SORT-style tracker

### Root `crash_signals.py`

Path: [crash_signals.py](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/crash_signals.py)

This file is a compatibility shim for older imports. New code should prefer:

```python
from src.accidents_cvpr.crash_signals import ...
```

## Artifacts And Outputs

Current artifact locations:

- `outputs/`: plots, rendered clips, and batch-analysis CSVs
- `checkpoints/`: trained model checkpoints
- `weights/`: local pretrained model weights
- `submissions/`: generated submission CSVs
- `archive/`: large archived inputs such as the local competition zip

Examples currently present:

- [outputs/yolo26_batch_results.csv](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/outputs/yolo26_batch_results.csv)
- [outputs/yolo26_score_dist.png](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/outputs/yolo26_score_dist.png)
- [checkpoints/best_model.pt](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/checkpoints/best_model.pt)
- [submissions/submission.csv](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/submissions/submission.csv)

## Environment

Current local configuration:

- Python version: `3.13` from [`.python-version`](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/.python-version)
- dependency manifest: [pyproject.toml](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/pyproject.toml)
- lockfile: [uv.lock](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/uv.lock)

Install the base environment with `uv`:

```bash
uv sync
```

The current `pyproject.toml` only covers the lightweight shared Python dependencies. Notebook-specific deep learning dependencies may still need to be installed separately depending on which experiment you run.

## Working Workflow

### If you want the current practical path

Use the YOLO26 notebook first:

- inspect or tune frame-sampling inference
- generate diagnostics into `outputs/`
- write submission files into `submissions/`

### If you want crash-signal research

Use the SAM 3.1 notebook plus the reusable module:

- detector output drives tracked objects
- `crash_signals.py` provides velocity, flow, and heading-based signals
- use this path for qualitative debugging and fusion ideas

### If you want temporal model baselines

Use the R2Plus1D or R3D notebooks:

- train on synthetic videos
- evaluate with the competition-style objective
- export submission-shaped outputs

## Important Notes

- `main.py` is currently placeholder-level and is not the canonical project entrypoint.
- The repo is still notebook-heavy. The `src/` directory is only the start of extracting stable reusable logic.
- This workspace intentionally keeps large datasets and model artifacts local rather than trying to version them in git.
- The repo may contain exploratory material in [NOTES.md](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/NOTES.md) and tool guidance in [AGENTS.md](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/AGENTS.md) and [CLAUDE.md](/home/andrewgirgis/Downloads/kaggle/accidents-cvpr/CLAUDE.md).

## Git Hygiene

The `.gitignore` is set up to exclude:

- large datasets
- local model artifacts
- generated outputs
- local archives
- virtual environments and notebook checkpoints

If a file is expensive to regenerate but not appropriate for source control, it should live in one of the local artifact folders rather than the repo root.

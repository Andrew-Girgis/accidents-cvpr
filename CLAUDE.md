# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

**CVPR 2026 ACCIDENT** — Kaggle competition (deadline: April 10, 2026) for traffic accident understanding in CCTV video. The task is to predict three things for each test video:

1. **`accident_time`** — When the accident occurs (seconds, float)
2. **`center_x, center_y`** — Where in the frame it occurs (normalized 0–1 coordinates)
3. **`type`** — Collision type (`head-on`, `rear-end`, `sideswipe`, `t-bone`, `single`)

Evaluation is the **harmonic mean** of three Gaussian-similarity scores (temporal T, spatial S, classification accuracy C).

## Key Challenge

**Zero-shot domain generalization**: Only synthetic training data is provided (CARLA simulator output). Test set is real CCTV footage — no real labeled training examples exist. Models must generalize from sim-to-real without fine-tuning on real labels.

## Data Structure

```
sim_dataset/
  labels.csv              — Ground truth for 2,211 synthetic training videos
  annotation_classes.yaml — 22 object classes (vehicles, pedestrians, infrastructure)
  videos/                 — Synthetic .mp4 files organized by collision type
  video_annotations/      — Per-frame JSON annotations (gzip-compressed)

videos/                   — 2,027 real test .mp4 files (hashed filenames)
test_metadata.csv         — Metadata for test videos (region, scene, weather, quality)
kaggle_comp.md            — Full competition spec and evaluation formula
```

## Key Data Details

**`sim_dataset/labels.csv` columns:**
- `rgb_path`, `annotations_path` — paths to video and annotation files
- `type`, `accident_time`, `accident_frame` — ground truth targets
- `center_x`, `center_y`, `x1`, `y1`, `x2`, `y2` — impact location (normalized)
- `map` — virtual environment (`Town03`–`Town10HD`)
- `weather` — `clear`, `night`, `rain`, `sunset`, `wet`
- `camera_position` — camera angle ID
- `no_frames`, `duration`, `height`, `width`

**`test_metadata.csv` columns:**
- `path` — video filename
- `region` — 20+ geographic regions (US states, UAE, World, etc.)
- `scene_layout` — `highway`, `signalized_intersection`, `city_street`, `grade_separated_intersection`, `simple_intersection`, `parking_lot`, `roundabout`, `tunnel`
- `weather` — `normal`, `rain`, `snow`
- `day_time` — `day` or `night`
- `quality` — `Excellent`, `Fine`, `Good`, `Poor`, `Very_Poor`

**Per-frame annotation JSON structure** (in `video_annotations/`):
- Keys: `base`, `collision`, `sensor`
- Each frame: `iteration` (frame number), `objects` array
- Each object: `id`, `tag` (class ID), 3D `location`/`extent`/`rotation`, 2D bounding box

## Submission Format

CSV with columns: `path, accident_time, center_x, center_y, type`

## Evaluation Metric

```
score = harmonic_mean(T, S, C)

T = exp(-((t_pred - t_true)^2) / (2 * sigma_t^2))   # temporal similarity
S = exp(-((dx^2 + dy^2) / (2 * sigma_s^2)))           # spatial similarity
C = 1 if type_pred == type_true else 0                 # classification accuracy
```

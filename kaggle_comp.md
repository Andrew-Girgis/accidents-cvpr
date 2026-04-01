# Overview

This challenge focuses on understanding real traffic car accidents in fixed-view CCTV video. Given a clip, your goal is to predict when the accident happens, where it happens in the frame, and what type of collision it is. In other words, this competition combines temporal localization + spatial localization + accident-type classification. Submissions are evaluated with a single leaderboard score that reflects performance across all three predictions: time, location, and collision type.

Unlike most Kaggle competitions, we don't provide real labeled training data. The benchmark is meant to evaluate how well your method works out of the box on real CCTV accidents. To help you build and test your pipeline, we provide a synthetic training set you can use for pretraining and debugging.

**Start:** 2 months ago | **Close:** 10 days to go

## Timeline

- **10 Feb 2026:** Competition Start
- **10 April 2026:** Competition Deadline
- **1 May 2026:** Deadline for paper submission to AUTOPILOT [Non-Archival Track]
- **7 May 2026:** Notification of acceptance / rejection
- **2-3 June 2026:** CVPR in Denver - Colorado [USA]

All deadlines are at 11:59 PM UTC of the corresponding day unless otherwise stated.

The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Challenge Description

### Motivation

Traffic accidents are rare but high-impact events. Being able to automatically understand what happened and where it happened can support faster incident response, better safety analytics, and more reliable reporting—especially in places where fixed CCTV cameras are the main source of coverage. But CCTV video is challenging: crashes can be far from the camera, partially blocked, low-resolution, and visually cluttered, which makes manual review slow and expensive.

Most video benchmarks assume you can train on a large labeled dataset from the same source. Real deployments often don't have that luxury. Labels are hard to get, rules vary by location, and camera setups change over time. That's why this competition is set up differently: we do not provide a labeled real training split. The goal is to test methods that generalize well without tuning on an official real training set.

To make the task practical, we provide synthetic data that you can use to prototype your pipeline, pretrain models, or validate ideas before running on the real test clips.

### What will you learn?

You'll learn how to build a pipeline for CCTV accident understanding that predicts three things from each clip:

- when the accident happens (time),
- where it happens in the frame (location),
- and what type of collision it is.

You'll also learn how to design and evaluate models in a training-free / zero-shot setting, where success depends on robustness and generalization rather than dataset-specific fitting.

## Evaluation

We score solutions on the three things you predict for each clip: accident time, impact location, and collision type. Each part produces a score in [0, 1] (higher is better), and the final leaderboard score is their harmonic mean (so doing poorly on any one part hurts the overall score).

### Component Scores

- **T (Temporal score)** — measures how close your predicted accident time is to the ground-truth time. It uses a Gaussian-style similarity: small time errors score near 1, larger errors drop smoothly toward 0.

- **S (Spatial score)** — measures how close your predicted impact point (x, y) is to the ground-truth location in the frame. It also uses a Gaussian-style similarity: small location errors score near 1, larger errors drop toward 0.

- **C (Classification score)** — Top-1 accuracy for collision type (1 if the predicted type matches, 0 otherwise; averaged across videos).

### Final Score (Harmonic Mean)

The final leaderboard score is the harmonic mean of **T**, **S**, and **C**.

### Submission File

Submit a CSV with a header and one row per test video:

```csv
path, accident_time, center_x, center_y, type
videos/iKpoAkiKqjw_00.mp4, 8.25, 0.633, 0.125, t-bone
videos/dQ0ao_Yu7II_00.mp4, 19.79, 0.197, 0.892, head-on
videos/nFpT2-RpAS8_2_00.mp4, 6.01, 0.529, 0.985, rear-end
videos/9AnFK85qRjM_00.mp4, 3.87, 0.847, 0.005, head-on
...
```

## Organizers and Contributors

- Lukas Picek, PiVa AI / University of West Bohemia in Pilsen / MIT [lukaspicek@gmail.com]
- Vojtěch Čermák, Czech Technical University in Prague
- Marek Hanzl, University of West Bohemia in Pilsen
- Michal Čermák, Charles University in Prague

## CVPR Context

This competition is organized in connection with the AUTOPILOT workshop at CVPR 2026, which brings together work on safety-critical autonomous driving, including robust perception and video understanding.

Participants are encouraged (but not required) to engage with the workshop community, e.g., by sharing techniques, lessons learned, or a short write-up of their solution. Top-performing approaches may be highlighted as part of the workshop program. For workshop updates, submission details, and deadlines, please refer to the AUTOPILOT workshop page or as in the Kaggle Discussion.

## Citation

Lukas Picek, Vojtěch Čermák, Marek Hanzl, Michal Čermák. ACCIDENT @ CVPR. https://kaggle.com/competitions/accident, 2026. Kaggle.

## Dataset Description

The competition is built around real traffic accidents recorded by fixed CCTV cameras. The benchmark is intentionally training-free on real footage: there is no labeled real training split. The real clips are used for evaluation.

To help you develop and test ideas anyway, we also provide a synthetic (simulated) dataset generated from CCTV-style viewpoints, with automatic annotations.

### Test Set — Real CCTV Clips

The real set is assembled from public traffic-camera feeds (including aggregator channels). Clips are curated to remove edited, duplicate, or unusable footage while preserving the typical challenges of CCTV video—low resolution, compression artifacts, occlusions, and wide fields of view.

All relevant information is provided in `test_metadata.csv`, which lists the test videos along with a few coarse scene tags (e.g., lighting, weather, layout). These tags are meant to help with analysis (for example, checking whether performance drops at night), but they are not used for scoring.

### Synthetic CCTV-style Clips (Development Support)

Because there is no labeled real training data (real accidents are rare, and labeling CCTV reliably is expensive), the competition is set up as a training-free / zero-shot benchmark on real footage. To keep it practical to work on, we provide a CARLA-based synthetic dataset that mimics fixed-camera (CCTV-style) accident videos—so you can train, debug, and iterate your pipeline before running it on the real test clips.

#### What’s included in the synthetic set?

**1) An index with labels (`labels.csv`)**

This is the “table of contents” for the synthetic dataset. Each row corresponds to one synthetic video and includes:

- `rgb_path` — path to the RGB video clip
- `type` — collision type label (string)
- `accident_time` and `accident_frame` — when the crash happens (in seconds and as a frame index)
- `center_x`, `center_y` — impact point in the frame (normalized to [0, 1])
- `x1`, `y1`, `x2`, `y2` — a normalized bounding box around the crash participants at impact
- scenario metadata like `map`, `weather`, `camera_position`, plus `duration`, `no_frames`, `height`, `width`
- `annotations_path` — path to a per-video annotation file (`.json.gz`)
- `annotations_start_offset` — an offset used to align timestamps when reading the annotation file

**2) Per-video annotation files (`*.json.gz`)**

These contain richer, frame-level information exported from the simulator (useful if you want more supervision than just a single time/location/type label). Think of them as “extra signals” you can optionally use for training or analysis.

**3) A class map for segmentation (`annotation_classes.yaml`)**

This maps numeric IDs to semantic classes used in the synthetic segmentation output. Examples include: Roads, SideWalks, Building, TrafficLight, TrafficSign, and dynamic agents like Pedestrian, Car, Truck, Bus, Motorcycle, Bicycle, etc.

> **Important note:** The synthetic set is provided only to support development. Your final score is computed on the real CCTV test set.

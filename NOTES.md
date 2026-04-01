# CVPR 2026 Accident Detection — Implementation Notes

## Competition Context
- Predict: `accident_time`, `center_x/y` (normalised), `collision_type` per clip
- Score: harmonic mean of T (temporal Gaussian σ=3s), S (spatial Gaussian), C (type accuracy)
- No labelled real training data — must generalise zero-shot
- Synthetic training set: 2,211 CARLA clips, 5 towns, 4 types, 5 weathers

---

## Why Bounding Box IoU Failed
- Cars share road space constantly — side-by-side triggers false positives
- No temporal context — can't distinguish passing from collision
- Proximity is necessary but not sufficient for crash detection

---

## New Approach: Physics + Trajectory Signals

Three complementary signals, fused by voting:

### Signal 1 — Track-Based Velocity Drop
**Idea:** Track each vehicle across frames (SORT). Compute rolling velocity (px/frame). 
A crash = sudden deceleration in ≥1 track (>50% drop in <5 frames).

**Why it works:** Passing cars maintain speed. Crashed cars stop.

**Libraries:**
- Detector: SAM 3.1 (current) or YOLOv8 (faster)
- Tracker: SORT (simple Kalman filter + Hungarian matching)
  - pip install `filterpy` + `lapjv` or use `ultralytics` built-in tracking

**Key parameters to tune (from GT analysis):**
- `vel_drop_thresh` — % velocity drop to trigger
- `vel_window` — frames over which to compute rolling velocity
- `decel_window` — frames over which to detect the drop

**GT validation:** Synthetic JSON gives box positions per iteration.
Compute centre velocity per object ID, plot around collision frames.

---

### Signal 2 — Optical Flow Anomaly
**Idea:** Dense optical flow (Farneback, built into OpenCV) between consecutive frames.
In the region where two close tracks overlap, compute:
- Mean flow magnitude
- Flow vector coherence (std of angles)

**Crash signal:** High magnitude (vehicles moving) followed by near-zero magnitude 
(vehicle stopped) OR chaotic angles (impact, spinning, deformation).

**Zero dependencies beyond OpenCV.**

**Key metrics:**
- `flow_magnitude_drop` — region went from moving to near-stationary
- `flow_angle_std` — coherence breakdown (chaos)

---

### Signal 3 — Trajectory Intersection + Heading
**Idea:** For each tracked vehicle, fit a linear trajectory to the last N centres.
Project forward. Check if paths geometrically intersect.
Also compute heading angle (atan2 of velocity vector).

**Crash type classification from headings:**
| Heading angle difference | Type |
|--------------------------|------|
| ~180° (opposite) | head-on |
| ~90° (perpendicular) | t-bone |
| ~0° / ~180° + offset | rear-end / sideswipe |

**This directly addresses the classification score C.**

---

## Fusion Strategy
Simple voting: crash if ≥2 of 3 signals agree in the same ±N frame window.
- Reduces false positives from Signal 1 alone (passing cars)
- Provides redundancy when tracker loses ID

---

## Implementation Plan

### Phase 0 — GT Signal Analysis (no model needed)
1. Load all `video_annotations/*.json` (or `.json.gz`)
2. For each object ID, compute frame-to-frame velocity using GT boxes
3. Plot velocity curves around `collision[0].iteration`
4. Find threshold separating crash deceleration from normal traffic

### Phase 1 — SORT Tracker Module
1. Implement/install SORT with a clean interface: `track(boxes, scores) → [(id, box), ...]`
2. Maintain per-ID velocity history (rolling mean of centre displacement)
3. Detect velocity drop events

### Phase 2 — Optical Flow Module
1. Compute Farneback flow between consecutive frames
2. For each active track pair within proximity, extract flow in overlap region
3. Compute magnitude + coherence signals

### Phase 3 — Trajectory Module
1. Maintain per-ID trajectory history (last 10 centres)
2. Fit linear regression to trajectory → heading vector
3. Project forward → find intersection point and ETA
4. Classify heading angle difference → collision type

### Phase 4 — Fusion + Integration into Notebook
1. Replace IoU crash detection with fused signal
2. Update draw_frame to show velocity/flow overlays
3. Update batch `process_clip` with new pipeline

---

## GT Signal Analysis Results (2026-03-31)

### Key Finding: Velocity Drop is a STRONG signal
All crashing vehicles drop to ~0 velocity post-crash. But naive threshold fails
because many non-crashing vehicles are already stationary (MaxVel=0 throughout).

**Discriminator: vehicle must have been MOVING before crash, then STOP.**

| Collision Type | Crash IDs MaxVel | MinVelPost | Non-crash false positives |
|----------------|-------------------|------------|---------------------------|
| head-on | 15.55 → 0.00 | 0 | several stationary bg cars also 100% drop |
| t-bone | 6.88, 5.79 → 0.00 | 0 | some bg cars also 100% drop |
| rear-end | 16.67, 8.09 → 0.00 | 0 | few non-crash false positives |
| sideswipe | 7.99, 7.89 → ~0.00 | 0 | almost none — very clean |

**Threshold that works:**
```
crash_vehicle = (MaxVel_pre > 3.0 px/iter) AND (MinVel_post < 0.5 px/iter)
```
This filters out stationary background vehicles (MaxVel ≈ 0) while catching all
crashing vehicles. True positive rate: 100% on 4 sample clips.

### False Positive Problem
Some non-crashing vehicles in the scene also stop (e.g. at traffic lights, yielding).
These would be missed by GT boxes (which have consistent IDs) but the tracker may
lose IDs. Need Signal 2 (optical flow) and Signal 3 (trajectory) to disambiguate.

### Heading Analysis: Pre-computing from GT
Object `location` field in JSON is 3D world-space. The `rotation.yaw` field gives
us the vehicle's actual heading angle — use this to validate Signal 3.

---

## Research Agent Findings (2026-03-31)

### Signal 1 — SORT Tracker

**Best option for raw numpy boxes:** vendor `abewley/sort.py` (single file) or our `SORTLite` (already implemented — no external deps).

Key parameters for overhead CCTV (research finding):
- `iou_threshold=0.15` (lower than default 0.3 — overhead boxes are smaller)
- `max_age=10` (bridge occlusions from signs/trees)

Velocity extraction: **compute from centroid history** (not Kalman internals) — more portable. Already done in `VelocityMonitor`.

ByteTrack vs SORT: ByteTrack wins in dense traffic (two-round Hungarian matching rescues low-conf detections). Can enable via `ultralytics.trackers.BYTETracker` if ID switches become a problem.

### Signal 2 — Optical Flow

**DIS MEDIUM** (`cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)`) is the right choice:
- ~10–15 fps at 1080p vs 2–3 fps for Farneback
- Equivalent quality; handles large displacements better

Farneback params updated for 1080p: `levels=5, winsize=21, poly_n=7, poly_sigma=1.5`

Crash signal (Wang et al. 2023, IEEE T-ITS): **spike-then-drop pattern**:
1. Pre-impact: rising magnitude as vehicles approach
2. Post-impact: >50% drop within 3–5 frames + angle coherence collapse

Angle coherence: use **circular mean resultant length** (Mardia & Jupp), not std-dev. Already updated in code.

### Signal 3 — Trajectory / Heading

Angular thresholds (NHTSA typology + Wang et al. 2021 IEEE T-ITS):

| Type | Heading Diff | Lateral Offset |
|------|-------------|----------------|
| rear-end | 0–30° | < threshold |
| sideswipe | 0–25° | ≥ threshold |
| t-bone | 70–110° | any |
| head-on | 150–180° | < threshold |
| sideswipe (opp) | 150–180° | ≥ threshold |

Head-on vs opposite sideswipe: disambiguate by lateral offset at closest approach point.

**SVD heading estimation** (already implemented) is the correct approach per literature. N=8–12 frames at 20fps works well.

Kalman velocity state = better heading when frame rate is low or there are occlusions.

### Relevant Public Datasets
- **METEOR** (IIT Bombay, 2023) — Indian intersections, CCTV, collision type labels — public
- **inD Dataset** (Aachen) — top-down drone, gold-standard trajectories (no crash labels)
- **AI City Challenge 2022–24** — CCTV crash type labels (requires registration)
- **DAIR-V2X** (2022) — V2X sensor trajectories, Chinese intersections

---

## Progress Log

| Date | Action |
|------|--------|
| 2026-03-31 | Initial SAM 3.1 baseline with IoU crash detection — detected false positives |
| 2026-03-31 | Pivoting to physics-based signals: velocity drop + optical flow + trajectory |

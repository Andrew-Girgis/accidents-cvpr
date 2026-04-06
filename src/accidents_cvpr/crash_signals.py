"""
crash_signals.py
================
Three complementary crash-detection signals for the CVPR 2026 Accident challenge.

Signal 1 — Track-Based Velocity Drop (SORT-lite tracker + Kalman state)
Signal 2 — Optical Flow Anomaly (Farneback dense flow, OpenCV)
Signal 3 — Trajectory Intersection + Heading Classifier (collision type)

Fusion: crash if ≥2 of 3 signals agree within a ±FUSION_WINDOW frame window.
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Tuneable parameters  (validated against GT velocity analysis 2026-03-31)
# ─────────────────────────────────────────────────────────────────────────────

# Signal 1 — velocity
VEL_MIN_MOVING   = 3.0    # px/frame — vehicle must have been moving before crash
VEL_STOP_THRESH  = 0.8    # px/frame — "stopped" threshold after impact
VEL_WINDOW       = 10     # frames of history for rolling velocity
VEL_DROP_FRAMES  = 8      # frames over which to measure the drop

# Signal 2 — optical flow
# Research (Wang et al. 2023, IEEE T-ITS): DIS MEDIUM is faster + equivalent quality
# Farneback params: levels=5, winsize=21, poly_n=7, poly_sigma=1.5 for 1080p vehicle motion
FLOW_WIN_SIZE    = 21     # Farneback winsize — larger for vehicle blobs at 1080p
FLOW_POLY_N      = 7      # poly_n=7 for large textureless car roofs
FLOW_POLY_SIGMA  = 1.5    # paired with poly_n=7
FLOW_LEVELS      = 5      # more levels needed at 1080p for >30px/frame displacements
FLOW_MAG_DROP    = 0.50   # Wang et al. 2023: >50% drop in 3–5 frames is the threshold
FLOW_CHAOS_STD   = 0.8    # radian std-dev of flow angles → "chaotic" post-impact
FLOW_HISTORY     = 8      # frames — long enough to capture spike-then-drop pattern
USE_DIS_FLOW     = True   # use DISOpticalFlow (faster) instead of Farneback

# Signal 3 — trajectory / heading
TRAJ_HISTORY     = 12     # frames of centroid history for heading estimation
TTC_MAX_FRAMES   = 30     # max frames ahead to project for Time-to-Collision
TTC_DIST_THRESH  = 80     # px — paths "intersect" if projected pts come within this
HEADING_REAR_MAX = 30     # degrees — rear-end threshold
HEADING_SIDE_MAX = 25     # degrees — sideswipe threshold
HEADING_TBONE_LO = 60     # degrees — t-bone lower bound
HEADING_TBONE_HI = 120    # degrees — t-bone upper bound
# head-on: 150–180°; lateral offset threshold is in px relative to vehicle size

# Minimum track age before trajectory/flow signals can fire
# Prevents early false positives when vehicles are already close at scene start
MIN_TRACK_AGE    = 15     # frames — both tracks must have this many centroid history entries

# Fusion
FUSION_WINDOW    = 10     # frames — signals must agree within this window
MIN_SIGNALS      = 2      # need ≥ this many signals to declare crash


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def box_centroid(box: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → (cx, cy)"""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def nms(boxes: np.ndarray, scores: np.ndarray,
        masks: "np.ndarray | None" = None,
        iou_thresh: float = 0.5
        ) -> "tuple[np.ndarray, np.ndarray, np.ndarray | None]":
    """Non-maximum suppression — removes duplicate detections."""
    if len(scores) == 0:
        return boxes, scores, masks
    order = np.argsort(-scores)
    keep = []
    for i in order:
        discard = False
        for j in keep:
            x1 = max(boxes[i][0], boxes[j][0]); y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2]); y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ai = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
            aj = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            if inter / max(ai + aj - inter, 1e-6) > iou_thresh:
                discard = True; break
        if not discard:
            keep.append(i)
    m = masks[keep] if masks is not None else None
    return boxes[keep], scores[keep], m


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SORT-lite: minimal Kalman tracker (no external deps beyond numpy)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalmanTrack:
    """Single-object Kalman filter tracking [cx, cy, w, h, dx, dy, dw, dh]."""
    track_id: int
    hit_streak: int = 0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    # State: [cx, cy, w, h, dx, dy, dw, dh]
    x: np.ndarray = field(default_factory=lambda: np.zeros(8))
    P: np.ndarray = field(default_factory=lambda: np.eye(8) * 10)

    # Kalman matrices (constant velocity model)
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1],
    ], dtype=float))

    H: np.ndarray = field(default_factory=lambda: np.eye(4, 8))
    R: np.ndarray = field(default_factory=lambda: np.eye(4) * 1.0)
    Q: np.ndarray = field(default_factory=lambda: np.eye(8) * 0.01)

    def __post_init__(self):
        self.P[4:, 4:] *= 1000  # high uncertainty on initial velocity

    @classmethod
    def from_box(cls, box: np.ndarray, track_id: int) -> "KalmanTrack":
        t = cls(track_id=track_id)
        cx, cy = box_centroid(box)
        w, h = box[2] - box[0], box[3] - box[1]
        t.x[:4] = [cx, cy, w, h]
        return t

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._to_box()

    def update(self, box: np.ndarray):
        cx, cy = box_centroid(box)
        w, h = box[2] - box[0], box[3] - box[1]
        z = np.array([cx, cy, w, h])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def _to_box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @property
    def velocity(self) -> float:
        """Speed in pixels/frame from Kalman velocity state."""
        dx, dy = self.x[4], self.x[5]
        return math.sqrt(dx*dx + dy*dy)

    @property
    def heading_deg(self) -> float:
        """Heading in [0,360) degrees from Kalman velocity state."""
        dx, dy = self.x[4], self.x[5]
        return math.degrees(math.atan2(dy, dx)) % 360

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.x[0], self.x[1]])


class SORTLite:
    """
    Minimal SORT tracker using our KalmanTrack.
    No external dependencies — pure numpy.

    Usage:
        tracker = SORTLite()
        # each frame:
        tracked = tracker.update(boxes_Nx4, scores_N)
        # returns list of (track_id, box [x1,y1,x2,y2], KalmanTrack)
    """

    def __init__(self, max_age: int = 5, min_hits: int = 2, iou_thresh: float = 0.25):
        self.max_age   = max_age
        self.min_hits  = min_hits
        self.iou_thresh = iou_thresh
        self.tracks: list[KalmanTrack] = []
        self._next_id  = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray
               ) -> list[tuple[int, np.ndarray, KalmanTrack]]:
        """
        boxes  : (N, 4) float32  [x1,y1,x2,y2]
        scores : (N,)   float32
        Returns list of (track_id, predicted_box, track_object) for confirmed tracks.
        """
        # 1. Predict existing tracks
        predicted = [(t, t.predict()) for t in self.tracks]

        # 2. Hungarian matching via greedy IoU
        unmatched_dets = list(range(len(boxes)))
        unmatched_trks = list(range(len(predicted)))
        matched_pairs  = []

        if len(predicted) > 0 and len(boxes) > 0:
            iou_matrix = np.zeros((len(boxes), len(predicted)))
            for d, box in enumerate(boxes):
                for t_idx, (_, pred_box) in enumerate(predicted):
                    iou_matrix[d, t_idx] = box_iou(box, pred_box)

            # Greedy: sort by descending IoU
            pairs = np.array(np.unravel_index(
                np.argsort(-iou_matrix, axis=None), iou_matrix.shape)).T
            used_d, used_t = set(), set()
            for d, t_idx in pairs:
                if iou_matrix[d, t_idx] < self.iou_thresh:
                    break
                if d in used_d or t_idx in used_t:
                    continue
                matched_pairs.append((d, t_idx))
                used_d.add(d); used_t.add(t_idx)

            unmatched_dets = [d for d in range(len(boxes)) if d not in used_d]
            unmatched_trks = [t for t in range(len(predicted)) if t not in used_t]

        # 3. Update matched tracks
        for d, t_idx in matched_pairs:
            predicted[t_idx][0].update(boxes[d])

        # 4. Create new tracks for unmatched detections
        for d in unmatched_dets:
            self.tracks.append(KalmanTrack.from_box(boxes[d], self._next_id))
            self._next_id += 1

        # 5. Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. Return confirmed tracks only
        results = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                results.append((t.track_id, t._to_box(), t))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1 — Velocity Drop
# ─────────────────────────────────────────────────────────────────────────────

class VelocityMonitor:
    """
    Maintains a rolling velocity history per track ID.
    Triggers when a vehicle that was moving suddenly stops.
    """

    def __init__(self):
        # track_id → deque of (frame_idx, speed_px_per_frame)
        self._history: dict[int, deque] = {}
        # Diagnostic: filled every frame, keyed by track_id
        self.last_metrics: dict[int, dict] = {}

    def update(self, frame_idx: int, tracks: list[tuple[int, np.ndarray, KalmanTrack]]
               ) -> Optional[tuple[bool, list[int], np.ndarray]]:
        """
        Returns (crash_detected, [crashing_track_ids], crash_region_box) or None.
        """
        for tid, box, ktrack in tracks:
            spd = ktrack.velocity
            if tid not in self._history:
                self._history[tid] = deque(maxlen=VEL_WINDOW + VEL_DROP_FRAMES)
            self._history[tid].append((frame_idx, spd))

        # Check each track for sudden decel
        self.last_metrics = {}
        crash_ids = []
        crash_boxes = []
        for tid, box, ktrack in tracks:
            hist = list(self._history.get(tid, []))
            if len(hist) < VEL_DROP_FRAMES + 2:
                self.last_metrics[tid] = {'speed': ktrack.velocity, 'max_before': None, 'min_after': None}
                continue
            # Split into "before" and "after" windows
            before = [v for _, v in hist[:-VEL_DROP_FRAMES]]
            after  = [v for _, v in hist[-VEL_DROP_FRAMES:]]
            if not before:
                continue
            max_before = max(before)
            min_after  = min(after)
            self.last_metrics[tid] = {'speed': ktrack.velocity, 'max_before': max_before, 'min_after': min_after}
            # Must have been moving, then stopped
            if max_before >= VEL_MIN_MOVING and min_after <= VEL_STOP_THRESH:
                crash_ids.append(tid)
                crash_boxes.append(box)

        if len(crash_ids) >= 1:
            # Merge boxes of crashing vehicles
            all_boxes = np.array(crash_boxes)
            merged = np.array([
                all_boxes[:, 0].min(), all_boxes[:, 1].min(),
                all_boxes[:, 2].max(), all_boxes[:, 3].max(),
            ])
            return True, crash_ids, merged

        return None

    def get_speed(self, track_id: int) -> float:
        hist = list(self._history.get(track_id, []))
        if not hist:
            return 0.0
        return hist[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — Optical Flow Anomaly
# ─────────────────────────────────────────────────────────────────────────────

class FlowAnomalyDetector:
    """
    Computes dense Farneback optical flow between consecutive frames.
    In each active track pair's overlap region, detects:
      - Magnitude drop (both vehicles stop suddenly)
      - Angular chaos (flow vectors become incoherent post-impact)
    """

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        # Stores (frame_idx, mean_magnitude) per region key
        self._region_mag_history: dict[str, deque] = {}
        # Diagnostic: filled every frame for the closest qualifying pair
        self.last_metrics: dict = {}

    def update(self, frame_bgr: np.ndarray, frame_idx: int,
               tracks: list[tuple[int, np.ndarray, KalmanTrack]]
               ) -> Optional[tuple[bool, np.ndarray, np.ndarray]]:
        """
        Returns (crash_detected, flow_vis_frame, crash_region_box) or None.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or len(tracks) < 2:
            self._prev_gray = gray
            return None

        # Compute dense flow — DIS MEDIUM preferred (faster, same quality at 1080p)
        if USE_DIS_FLOW:
            if not hasattr(self, '_dis'):
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = self._dis.calc(self._prev_gray, gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=FLOW_LEVELS, winsize=FLOW_WIN_SIZE,
                iterations=3, poly_n=FLOW_POLY_N, poly_sigma=FLOW_POLY_SIGMA,
                flags=0,
            )
        self._prev_gray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Check all close pairs of tracks
        self.last_metrics = {}
        result = None
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                tid_i, box_i, ktrack_i = tracks[i]
                tid_j, box_j, ktrack_j = tracks[j]

                # Require both tracks to have enough history before firing
                if ktrack_i.hits < MIN_TRACK_AGE or ktrack_j.hits < MIN_TRACK_AGE:
                    continue

                # Only check pairs whose boxes are within proximity
                cx_i, cy_i = box_centroid(box_i)
                cx_j, cy_j = box_centroid(box_j)
                dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                box_size = max(box_i[2]-box_i[0], box_i[3]-box_i[1],
                               box_j[2]-box_j[0], box_j[3]-box_j[1])
                if dist > box_size * 3:
                    continue  # too far apart

                # Union box of the pair
                union = np.array([
                    min(box_i[0], box_j[0]), min(box_i[1], box_j[1]),
                    max(box_i[2], box_j[2]), max(box_i[3], box_j[3]),
                ])
                x1, y1, x2, y2 = map(int, np.clip(union, 0,
                    [frame_bgr.shape[1]-1, frame_bgr.shape[0]-1]*2))
                if x2 <= x1 or y2 <= y1:
                    continue

                region_mag = mag[y1:y2, x1:x2]
                region_ang = ang[y1:y2, x1:x2]

                mean_mag = float(region_mag.mean())
                # Circular mean resultant length (Mardia & Jupp) — 1=coherent, 0=chaotic
                ux = np.cos(region_ang); uy = np.sin(region_ang)
                coherence = float(np.sqrt(np.mean(ux)**2 + np.mean(uy)**2))
                ang_std   = 1.0 - coherence   # invert: high value = chaotic

                key = f"{min(tid_i,tid_j)}_{max(tid_i,tid_j)}"
                if key not in self._region_mag_history:
                    self._region_mag_history[key] = deque(maxlen=FLOW_HISTORY + 3)
                self._region_mag_history[key].append((frame_idx, mean_mag))

                hist = list(self._region_mag_history[key])

                before_mags = [v for _, v in hist[:-3]] if len(hist) >= 3 else []
                after_mags  = [v for _, v in hist[-3:]] if len(hist) >= 1 else []
                max_before  = max(before_mags) if before_mags else 0.0
                min_after   = min(after_mags)  if after_mags  else 0.0
                mag_drop    = (max_before - min_after) / max(max_before, 0.1)

                # Always record metrics for the first qualifying pair (diagnostic)
                if not self.last_metrics:
                    self.last_metrics = {
                        'pair': (tid_i, tid_j),
                        'pair_dist': dist,
                        'mean_mag': mean_mag,
                        'max_before': max_before,
                        'min_after': min_after,
                        'mag_drop': mag_drop,
                        'ang_std': ang_std,
                        'hist_len': len(hist),
                    }

                if len(hist) < FLOW_HISTORY:
                    continue

                is_chaotic = ang_std > FLOW_CHAOS_STD

                if (mag_drop >= FLOW_MAG_DROP and max_before > 0.5) or is_chaotic:
                    result = (True, flow, union)
                    break
            if result:
                break

        return result

    def visualise_flow(self, flow: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
        """HSV colour-coded flow overlay for debugging."""
        h, w = flow.shape[:2]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.addWeighted(frame_bgr, 0.6,
                               cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.4, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — Trajectory Intersection + Heading Classifier
# ─────────────────────────────────────────────────────────────────────────────

COLLISION_TYPES = ("head-on", "t-bone", "rear-end", "sideswipe", "unknown")


def estimate_heading(centroids: np.ndarray) -> float:
    """
    Estimate heading from recent centroid history using SVD (PCA).
    Returns degrees in [0, 360) from +x axis.
    Resolves 180° ambiguity from displacement direction.
    """
    if len(centroids) < 2:
        return 0.0
    pts = centroids[-TRAJ_HISTORY:]
    centred = pts - pts.mean(axis=0)
    if np.allclose(centred, 0):
        return 0.0
    _, _, vt = np.linalg.svd(centred)
    direction = vt[0]
    displacement = pts[-1] - pts[0]
    if np.dot(direction, displacement) < 0:
        direction = -direction
    return math.degrees(math.atan2(direction[1], direction[0])) % 360


def classify_collision_type(heading_a: float, heading_b: float,
                             centroid_a: np.ndarray, centroid_b: np.ndarray) -> str:
    """
    Classify collision type from two vehicle headings and lateral offset.

    heading_a, heading_b: degrees [0, 360)
    centroid_a, centroid_b: (x, y) pixel positions
    """
    # Angular difference (0–180)
    diff = abs(heading_a - heading_b) % 360
    if diff > 180:
        diff = 360 - diff

    # Lateral offset between centroids (perpendicular to mean heading)
    mean_heading_rad = math.radians((heading_a + heading_b) / 2)
    offset_vec = centroid_b - centroid_a
    lateral = abs(offset_vec[0] * math.sin(mean_heading_rad)
                  - offset_vec[1] * math.cos(mean_heading_rad))

    if diff <= HEADING_REAR_MAX:
        # Same direction — rear-end or sideswipe
        if lateral < 40:
            return "rear-end"
        else:
            return "sideswipe"
    elif diff >= 150:
        # Opposing — head-on or opposite-direction sideswipe
        if lateral < 120:   # generous — vehicles are 100-200px wide in CCTV overhead view
            return "head-on"
        else:
            return "sideswipe"
    elif HEADING_TBONE_LO <= diff <= HEADING_TBONE_HI:
        return "t-bone"
    else:
        return "unknown"


def project_trajectory(centroids: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Linear projection of the last few centroids n_frames into the future.
    Returns predicted centroid position.
    """
    if len(centroids) < 2:
        return centroids[-1]
    recent = centroids[-min(6, len(centroids)):]
    # Fit linear trend
    t = np.arange(len(recent), dtype=float)
    vx = np.polyfit(t, recent[:, 0], 1)[0]
    vy = np.polyfit(t, recent[:, 1], 1)[0]
    return centroids[-1] + np.array([vx, vy]) * n_frames


class TrajectoryAnalyser:
    """
    Maintains centroid history per track.
    Detects trajectory intersection and classifies collision type.
    """

    def __init__(self):
        self._centroids: dict[int, deque] = {}  # track_id → deque of (x,y)
        # Diagnostic: filled every frame for the closest pair evaluated
        self.last_metrics: dict = {}

    def update(self, tracks: list[tuple[int, np.ndarray, KalmanTrack]]
               ) -> Optional[tuple[bool, str, np.ndarray, float]]:
        """
        Returns (crash_detected, collision_type, crash_box, confidence) or None.
        """
        for tid, box, _ in tracks:
            if tid not in self._centroids:
                self._centroids[tid] = deque(maxlen=TRAJ_HISTORY + 5)
            self._centroids[tid].append(box_centroid(box))

        # Check all pairs
        track_ids = [t[0] for t in tracks]
        boxes     = {t[0]: t[1] for t in tracks}

        self.last_metrics = {}
        best_result = None
        best_min_dist = float("inf")

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                tid_i = tracks[i][0]
                tid_j = tracks[j][0]

                hist_i = np.array(list(self._centroids.get(tid_i, [])))
                hist_j = np.array(list(self._centroids.get(tid_j, [])))

                # Always log hist length for the first pair (even if skipped)
                if not self.last_metrics:
                    self.last_metrics = {
                        'pair': (tid_i, tid_j),
                        'hist_len_i': len(hist_i),
                        'hist_len_j': len(hist_j),
                        'min_proj_dist': None,
                        'curr_dist': None,
                        'heading_i': None,
                        'heading_j': None,
                        'ctype': None,
                    }

                if len(hist_i) < MIN_TRACK_AGE or len(hist_j) < MIN_TRACK_AGE:
                    continue

                # Current distance
                curr_dist = float(np.linalg.norm(hist_i[-1] - hist_j[-1]))

                # Heading estimation
                heading_i = estimate_heading(hist_i)
                heading_j = estimate_heading(hist_j)

                # Project trajectories forward
                min_proj_dist = float("inf")
                for k in range(1, TTC_MAX_FRAMES + 1):
                    proj_i = project_trajectory(hist_i, k)
                    proj_j = project_trajectory(hist_j, k)
                    d = float(np.linalg.norm(proj_i - proj_j))
                    if d < min_proj_dist:
                        min_proj_dist = d

                # Update diagnostic with best pair info this frame
                if min_proj_dist < best_min_dist or not self.last_metrics.get('min_proj_dist'):
                    self.last_metrics.update({
                        'pair': (tid_i, tid_j),
                        'hist_len_i': len(hist_i),
                        'hist_len_j': len(hist_j),
                        'curr_dist': curr_dist,
                        'min_proj_dist': min_proj_dist,
                        'heading_i': heading_i,
                        'heading_j': heading_j,
                        'ctype': classify_collision_type(heading_i, heading_j, hist_i[-1], hist_j[-1]),
                    })

                # Trajectories converging to within threshold?
                if min_proj_dist < TTC_DIST_THRESH and min_proj_dist < best_min_dist:
                    ctype = classify_collision_type(
                        heading_i, heading_j, hist_i[-1], hist_j[-1]
                    )
                    merged = np.array([
                        min(boxes[tid_i][0], boxes[tid_j][0]),
                        min(boxes[tid_i][1], boxes[tid_j][1]),
                        max(boxes[tid_i][2], boxes[tid_j][2]),
                        max(boxes[tid_i][3], boxes[tid_j][3]),
                    ])
                    # Confidence: closer projected paths = higher confidence
                    conf = max(0.0, 1.0 - min_proj_dist / TTC_DIST_THRESH)
                    best_min_dist = min_proj_dist
                    best_result   = (True, ctype, merged, conf)

        return best_result

    def get_heading(self, track_id: int) -> float:
        hist = list(self._centroids.get(track_id, []))
        if not hist:
            return 0.0
        return estimate_heading(np.array(hist))


# ─────────────────────────────────────────────────────────────────────────────
# Fused Crash Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrashEvent:
    frame_idx: int
    crash_box: np.ndarray          # [x1,y1,x2,y2] in native frame coords
    collision_type: str            # head-on / t-bone / rear-end / sideswipe
    signals_triggered: list[str]   # which signals fired
    confidence: float              # 0–1


class FusedCrashDetector:
    """
    Combines all three signals. Emits a CrashEvent when ≥ MIN_SIGNALS agree
    within FUSION_WINDOW frames.
    """

    def __init__(self):
        self.tracker   = SORTLite()
        self.vel_mon   = VelocityMonitor()
        self.flow_det  = FlowAnomalyDetector()
        self.traj_anal = TrajectoryAnalyser()

        # Recent signal hits: (frame_idx, signal_name, box, ctype)
        self._signal_hits: deque = deque(maxlen=FUSION_WINDOW * 3)
        self.crash_events: list[CrashEvent] = []
        self._last_crash_frame: int = -FUSION_WINDOW * 2  # prevent duplicate events
        self._crashed: bool = False  # lock: emit at most 1 crash per clip
        # Per-frame diagnostic log — list of dicts, one per frame
        self.diag_log: list[dict] = []

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int,
                      det_boxes: np.ndarray, det_scores: np.ndarray
                      ) -> Optional[CrashEvent]:
        """
        Call every frame.

        det_boxes  : (N,4) float32 [x1,y1,x2,y2] — vehicle detections this frame
        det_scores : (N,)  float32 — confidence scores

        Returns CrashEvent if crash is detected, else None.
        """
        # Update tracker
        if len(det_boxes) > 0:
            tracks = self.tracker.update(det_boxes, det_scores)
        else:
            tracks = self.tracker.update(np.zeros((0, 4)), np.zeros(0))

        # --- Signal 1: velocity drop ---
        vel_result = self.vel_mon.update(frame_idx, tracks)
        if vel_result:
            _, _, vbox = vel_result
            self._signal_hits.append((frame_idx, "velocity", vbox, "unknown"))

        # --- Signal 2: optical flow anomaly ---
        flow_result = self.flow_det.update(frame_bgr, frame_idx, tracks)
        if flow_result:
            _, _, fbox = flow_result
            self._signal_hits.append((frame_idx, "flow", fbox, "unknown"))

        # --- Signal 3: trajectory intersection ---
        traj_result = self.traj_anal.update(tracks)
        if traj_result:
            _, ctype, tbox, conf = traj_result
            self._signal_hits.append((frame_idx, "trajectory", tbox, ctype))

        # --- Diagnostic log ---
        signals_this_frame = [sn for fi, sn, _, _ in self._signal_hits if fi == frame_idx]
        self.diag_log.append({
            'frame_idx':   frame_idx,
            'n_tracks':    len(tracks),
            'vel':         dict(self.vel_mon.last_metrics),
            'flow':        dict(self.flow_det.last_metrics),
            'traj':        dict(self.traj_anal.last_metrics),
            'signals':     signals_this_frame,
        })

        # --- Fusion ---
        if self._crashed:
            return None  # one crash per clip — prevent blinking re-triggers
        if frame_idx - self._last_crash_frame < FUSION_WINDOW:
            return None  # cooldown — don't fire duplicate events

        recent = [(fi, sn, box, ct) for fi, sn, box, ct in self._signal_hits
                  if frame_idx - FUSION_WINDOW <= fi <= frame_idx]

        if len(recent) < MIN_SIGNALS:
            return None

        # Check how many distinct signals fired
        signal_names = set(sn for _, sn, _, _ in recent)
        if len(signal_names) < MIN_SIGNALS:
            return None

        # Merge boxes from all recent hits
        all_boxes = np.array([box for _, _, box, _ in recent])
        merged = np.array([
            all_boxes[:, 0].min(), all_boxes[:, 1].min(),
            all_boxes[:, 2].max(), all_boxes[:, 3].max(),
        ])

        # Use trajectory's collision type if available, else unknown
        ctypes = [ct for _, _, _, ct in recent if ct != "unknown"]
        ctype  = ctypes[0] if ctypes else "unknown"

        confidence = len(signal_names) / 3.0

        event = CrashEvent(
            frame_idx=frame_idx,
            crash_box=merged,
            collision_type=ctype,
            signals_triggered=list(signal_names),
            confidence=confidence,
        )
        self.crash_events.append(event)
        self._last_crash_frame = frame_idx
        self._crashed = True  # lock: no more events for this clip
        return event

    def get_tracks(self) -> list[tuple[int, np.ndarray, KalmanTrack]]:
        """Return currently active tracks (for drawing)."""
        return [(t.track_id, t._to_box(), t) for t in self.tracker.tracks
                if t.time_since_update == 0]

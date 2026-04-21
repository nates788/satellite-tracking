from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    polygon: List[List[float]]
    confidence: float
    class_id: int
    image: str

    @property
    def aabb(self) -> np.ndarray:
        pts = np.array(self.polygon, dtype=float)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return np.array([x1, y1, x2, y2], dtype=float)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    class_id: int
    age: int = 0
    hits: int = 1
    history: List[Dict] = field(default_factory=list)

    def update(self, det: Detection):
        self.bbox = det.aabb
        self.age = 0
        self.hits += 1
        self.history.append(
            {
                "image": det.image,
                "bbox": self.bbox.tolist(),
                "confidence": det.confidence,
                "class_id": det.class_id,
            }
        )

    def step(self):
        self.age += 1


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.2, max_age: int = 5, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Detection]):
        for track in self.tracks:
            track.step()

        if not self.tracks:
            for det in detections:
                self._start_track(det)
            self._prune_tracks()
            return

        cost = np.ones((len(self.tracks), len(detections)), dtype=float)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                if track.class_id != det.class_id:
                    continue
                cost[i, j] = 1.0 - bbox_iou(track.bbox, det.aabb)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost[r, c]
            if iou >= self.iou_threshold:
                self.tracks[r].update(detections[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                self._start_track(det)

        self._prune_tracks()

    def _start_track(self, det: Detection):
        track = Track(track_id=self.next_id, bbox=det.aabb, class_id=det.class_id)
        track.history.append(
            {
                "image": det.image,
                "bbox": track.bbox.tolist(),
                "confidence": det.confidence,
                "class_id": det.class_id,
            }
        )
        self.tracks.append(track)
        self.next_id += 1

    def _prune_tracks(self):
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

    def export(self) -> List[Dict]:
        return [
            {
                "track_id": t.track_id,
                "class_id": t.class_id,
                "hits": t.hits,
                "age": t.age,
                "history": t.history,
            }
            for t in self.tracks
            if t.hits >= self.min_hits
        ]

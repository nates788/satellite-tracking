from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    frame_idx: int
    class_id: int
    conf: float
    cx: float
    cy: float
    w: float
    h: float
    angle: float


@dataclass
class Track:
    track_id: int
    class_id: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    angle: float = 0.0
    conf: float = 0.0
    age: int = 0
    hits: int = 1
    missed: int = 0

    def predict(self) -> None:
        self.cx += self.vx
        self.cy += self.vy
        self.age += 1
        self.missed += 1

    def update(self, det: Detection) -> None:
        new_vx = det.cx - self.cx
        new_vy = det.cy - self.cy
        self.vx = 0.7 * self.vx + 0.3 * new_vx
        self.vy = 0.7 * self.vy + 0.3 * new_vy
        self.cx = det.cx
        self.cy = det.cy
        self.w = det.w
        self.h = det.h
        self.angle = det.angle
        self.conf = det.conf
        self.hits += 1
        self.missed = 0


@dataclass
class FrameTracks:
    frame_idx: int
    frame_name: str
    tracks: List[Track]


def center_distance(det: Detection, track: Track) -> float:
    return math.hypot(det.cx - track.cx, det.cy - track.cy)


def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_radians: float) -> np.ndarray:
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(angle_radians)))
    return cv2.boxPoints(rect).astype(np.int32)


def draw_track(image: np.ndarray, track: Track) -> np.ndarray:
    corners = obb_to_corners(track.cx, track.cy, track.w, track.h, track.angle)
    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(
        image,
        f"ID {track.track_id}",
        (int(track.cx), int(track.cy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return image


def extract_detections_from_result(result, frame_idx: int) -> List[Detection]:
    detections: List[Detection] = []
    obb = getattr(result, "obb", None)
    if obb is None or obb.xywhr is None or len(obb.xywhr) == 0:
        return detections

    xywhr = obb.xywhr.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)
    confs = obb.conf.cpu().numpy()

    for row, class_id, conf in zip(xywhr, classes, confs):
        cx, cy, w, h, angle = row.tolist()
        detections.append(
            Detection(
                frame_idx=frame_idx,
                class_id=int(class_id),
                conf=float(conf),
                cx=float(cx),
                cy=float(cy),
                w=float(w),
                h=float(h),
                angle=float(angle),
            )
        )
    return detections


def greedy_association(
    detections: List[Detection],
    active_tracks: List[Track],
    max_distance: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for ti, track in enumerate(active_tracks):
        for di, det in enumerate(detections):
            if track.class_id != det.class_id:
                continue
            dist = center_distance(det, track)
            if dist <= max_distance:
                candidates.append((dist, ti, di))

    candidates.sort(key=lambda x: x[0])
    matches: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()

    for _, ti, di in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        matches.append((ti, di))
        used_tracks.add(ti)
        used_dets.add(di)

    unmatched_tracks = [i for i in range(len(active_tracks)) if i not in used_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def load_frame_paths(frames_dir: Path, limit: int | None = None) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    frame_paths = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if limit is not None:
        frame_paths = frame_paths[:limit]
    return frame_paths


def write_video_from_frames(frames_dir: Path, output_video: Path, fps: float) -> None:
    frame_paths = load_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No annotated frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Try mp4 first
    suffix = output_video.suffix.lower()

    if suffix == ".mp4":
        codecs = ["avc1", "mp4v"]
    elif suffix == ".avi":
        codecs = ["XVID", "MJPG"]
    else:
        codecs = ["mp4v"]

    writer = None

    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            print(f"Using codec: {codec}")
            break

    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_video}")

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {output_video}. "
            "Try a .avi output path or a different codec."
        )

    written = 0
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: skipping unreadable frame {frame_path}")
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        written += 1

    writer.release()

    if written == 0:
        raise RuntimeError("VideoWriter opened, but no frames were written.")

    if not output_video.exists():
        raise RuntimeError(f"Writer finished, but output video was not created: {output_video}")

    print(f"Wrote {written} frames to {output_video}")


def track_frame_sequence(
    model: YOLO,
    frames_dir: Path,
    output_frames_dir: Path,
    output_json: Path,
    output_video: Path | None,
    fps: float,
    imgsz: int,
    conf: float,
    max_distance: float,
    max_missed: int,
    limit: int | None,
) -> None:
    frame_paths = load_frame_paths(frames_dir, limit=limit)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    active_tracks: List[Track] = []
    all_frames: List[FrameTracks] = []
    next_track_id = 1

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        detections = extract_detections_from_result(result, frame_idx)

        for track in active_tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_dets = greedy_association(
            detections,
            active_tracks,
            max_distance=max_distance,
        )

        for ti, di in matches:
            active_tracks[ti].update(detections[di])

        for di in unmatched_dets:
            det = detections[di]
            active_tracks.append(
                Track(
                    track_id=next_track_id,
                    class_id=det.class_id,
                    cx=det.cx,
                    cy=det.cy,
                    w=det.w,
                    h=det.h,
                    angle=det.angle,
                    conf=det.conf,
                )
            )
            next_track_id += 1

        kept_tracks: List[Track] = []
        visible_tracks: List[Track] = []

        for track in active_tracks:
            if track.missed <= max_missed:
                kept_tracks.append(track)
            if track.missed <= max_missed and track.hits >= 2:
                visible_tracks.append(Track(**asdict(track)))

        active_tracks = kept_tracks

        annotated = frame.copy()
        for track in visible_tracks:
            annotated = draw_track(annotated, track)
        ok = cv2.imwrite(str(output_frames_dir / frame_path.name), annotated)
        if not ok:
            raise RuntimeError(f"Failed to save annotated frame: {output_frames_dir / frame_path.name}")

        all_frames.append(
            FrameTracks(
                frame_idx=frame_idx,
                frame_name=frame_path.name,
                tracks=visible_tracks,
            )
        )

    payload: Dict[str, Dict] = {
        str(item.frame_idx): {
            "frame_name": item.frame_name,
            "tracks": [asdict(track) for track in item.tracks],
        }
        for item in all_frames
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_video_from_frames(output_frames_dir, output_video, fps=fps)

    print(f"Saved annotated frames to {output_frames_dir}")
    print(f"Saved track JSON to {output_json}")
    print(f"Saved compiled video to {output_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects in a folder of satellite image frames using YOLO-OBB detections.")
    parser.add_argument("--model", required=True, help="Path to trained YOLO OBB weights.")
    parser.add_argument("--frames-dir", required=True, help="Path to directory containing ordered image frames.")
    parser.add_argument("--output-frames-dir", required=True, help="Directory to save annotated output frames.")
    parser.add_argument("--output-json", required=True, help="Path to save per-frame track JSON.")
    parser.add_argument("--output-video", required=True, help="Path to save compiled annotated MP4.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS to use when compiling annotated frames into video.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    track_frame_sequence(
        model=model,
        frames_dir=Path(args.frames_dir),
        output_frames_dir=Path(args.output_frames_dir),
        output_json=Path(args.output_json),
        output_video=Path(args.output_video),
        fps=args.fps,
        imgsz=args.imgsz,
        conf=args.conf,
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

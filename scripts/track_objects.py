import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.tracking import Detection, SimpleTracker


def load_predictions(pred_path: Path) -> Dict[str, List[dict]]:
    preds: Dict[str, List[dict]] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["image"]] = rec.get("detections", [])
    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects from exported OBB detections.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load_predictions(Path(args.predictions))
    sequences = json.loads(Path(args.sequences).read_text(encoding="utf-8"))

    tracker = SimpleTracker(
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )

    all_tracks = []
    for sequence in sequences:
        tracker = SimpleTracker(
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits,
        )
        for image_path in sequence:
            dets = []
            for d in predictions.get(image_path, []):
                dets.append(
                    Detection(
                        polygon=d["polygon"],
                        confidence=float(d["confidence"]),
                        class_id=int(d["class_id"]),
                        image=image_path,
                    )
                )
            tracker.update(dets)
        all_tracks.extend(tracker.export())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_tracks, indent=2), encoding="utf-8")
    print(f"Saved {len(all_tracks)} tracks to {out_path}")


if __name__ == "__main__":
    main()

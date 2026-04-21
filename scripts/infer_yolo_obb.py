import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO OBB inference and export detections.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        stream=True,
        verbose=False,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for result in results:
            boxes = []
            obb = getattr(result, "obb", None)
            if obb is not None and obb.xyxyxyxy is not None:
                polys = obb.xyxyxyxy.cpu().numpy().tolist()
                confs = obb.conf.cpu().numpy().tolist()
                clss = obb.cls.cpu().numpy().tolist()
                for poly, conf, cls_id in zip(polys, confs, clss):
                    boxes.append(
                        {
                            "polygon": poly,
                            "confidence": float(conf),
                            "class_id": int(cls_id),
                        }
                    )
            rec = {"image": str(result.path), "detections": boxes}
            f.write(json.dumps(rec) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

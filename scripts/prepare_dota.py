import argparse
import shutil
from pathlib import Path

from PIL import Image

from src.utils.geometry import dota_line_to_yolo_obb


DOTA_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_split(split_root: Path, output_root: Path, split_name: str):
    image_dir = split_root / "images"
    label_dir = split_root / "labelTxt"
    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"Skipping split '{split_name}': missing image dir {image_dir}")
        return

    if not label_dir.exists():
        print(f"Skipping split '{split_name}': missing label dir {label_dir}")
        return

    count = 0
    skipped_missing_image = 0
    skipped_bad_image = 0

    for label_file in sorted(label_dir.glob("*.txt")):
        image_path = find_image(image_dir, label_file.stem)
        if image_path is None:
            skipped_missing_image += 1
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Warning: could not open image {image_path}: {e}")
            skipped_bad_image += 1
            continue

        shutil.copy2(image_path, out_img_dir / image_path.name)

        yolo_lines = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()
            if lower.startswith("imagesource") or lower.startswith("gsd"):
                continue

            converted = dota_line_to_yolo_obb(
                stripped,
                img_w,
                img_h,
                CLASS_TO_ID,
            )
            if converted is not None:
                if isinstance(converted, str):
                    yolo_lines.append(converted)
                else:
                    yolo_lines.append(" ".join(map(str, converted)))

        out_label_path = out_lbl_dir / f"{label_file.stem}.txt"
        out_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        count += 1

    print(f"Converted {count} items for split '{split_name}'")
    if skipped_missing_image:
        print(f"  Skipped {skipped_missing_image} labels with no matching image")
    if skipped_bad_image:
        print(f"  Skipped {skipped_bad_image} unreadable images")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DOTA annotations to YOLO OBB format.")
    parser.add_argument("--dota-root", required=True, help="Root directory containing train/ and val/.")
    parser.add_argument("--output-root", required=True, help="Output directory for YOLO OBB data.")
    return parser.parse_args()


def main():
    args = parse_args()
    dota_root = Path(args.dota_root)
    output_root = Path(args.output_root)

    for split_name in ["train", "val"]:
        split_root = dota_root / split_name
        if split_root.exists():
            convert_split(split_root, output_root, split_name)
        else:
            print(f"Skipping split '{split_name}': missing directory {split_root}")

    print("Done. Update configs/dota8_obb.yaml with the new output path.")


if __name__ == "__main__":
    main()
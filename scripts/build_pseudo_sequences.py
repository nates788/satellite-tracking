import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build simple pseudo-sequences from an image directory.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

    sequences = []
    for i in range(0, len(image_paths), args.group_size):
        chunk = image_paths[i : i + args.group_size]
        if len(chunk) >= 2:
            sequences.append([str(p) for p in chunk])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sequences, indent=2), encoding="utf-8")
    print(f"Wrote {len(sequences)} sequences to {output}")


if __name__ == "__main__":
    main()

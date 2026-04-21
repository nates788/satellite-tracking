"""
Minimal extension point for a transformer-based baseline.

This file is intentionally lightweight. The idea is to keep the first stage of
this project focused on YOLOv8-OBB, then plug in a DETR-style model once the
baseline pipeline is stable.

Recommended directions:
- MMRotate + rotated DETR-style detector
- Deformable DETR adapted for aerial/remote sensing detection
- Any oriented-box transformer model compatible with DOTA
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DETR baseline entry point.")
    parser.add_argument("--config", required=False, help="Path to a DETR experiment config.")
    parser.add_argument("--data-root", required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    print("DETR baseline placeholder")
    print("Use this file as the handoff point once the YOLOv8-OBB baseline is complete.")
    print(f"config={args.config}")
    print(f"data_root={args.data_root}")


if __name__ == "__main__":
    main()

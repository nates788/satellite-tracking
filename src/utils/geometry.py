from __future__ import annotations

from typing import Iterable, List, Tuple


def dota_line_to_yolo_obb(line, img_w, img_h, class_map):
    parts = line.strip().split()

    # 8 coords + class + difficulty
    coords = list(map(float, parts[:8]))
    cls = parts[8]

    if cls not in class_map:
        return None

    cls_id = class_map[cls]

    # Normalize coordinates
    norm_coords = []
    for i, c in enumerate(coords):
        if i % 2 == 0:  # x
            norm_coords.append(c / img_w)
        else:  # y
            norm_coords.append(c / img_h)

    return [cls_id] + norm_coords


def polygon_flat_to_pairs(vals: Iterable[float]) -> List[Tuple[float, float]]:
    vals = list(vals)
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

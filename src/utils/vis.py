from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def draw_polygon(image: np.ndarray, polygon: List[List[float]], label: str | None = None) -> np.ndarray:
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    canvas = image.copy()
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    if label:
        x, y = pts[0, 0, 0], pts[0, 0, 1]
        cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

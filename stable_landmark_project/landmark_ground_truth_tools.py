"""
Utilities for turning hand-labeled side-by-side landmark screenshots into
machine-readable ground truth.

Expected folder layout:
    Corresponding Landmarks/
      Pair 1/
        baseline.png
        followup.png
        labeled_side_by_side.png

The widest image in each pair folder is treated as the labeled side-by-side
reference. The two narrower images are treated as image1 and image2 in filename
sort order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int


@dataclass
class BoxLabel:
    side: str
    color_key: str
    center: Point
    bbox: Tuple[int, int, int, int]
    area: int


def _read_image_info(path: Path) -> Optional[ImageInfo]:
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    return ImageInfo(path=path, width=w, height=h)


def discover_pair_folders(root: Path) -> List[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def discover_pair_images(folder: Path) -> Tuple[ImageInfo, ImageInfo, ImageInfo]:
    infos = []
    for path in sorted(folder.glob("*")):
        if path.name.startswith(".") or not path.is_file():
            continue
        info = _read_image_info(path)
        if info is not None:
            infos.append(info)

    if len(infos) < 3:
        raise ValueError(f"{folder}: expected at least 3 readable images")

    labeled = max(infos, key=lambda item: item.width)
    originals = [info for info in infos if info.path != labeled.path]
    originals = sorted(originals, key=lambda item: item.path.name)
    return originals[0], originals[1], labeled


def _dominant_color_key(bgr_pixels: np.ndarray) -> str:
    if bgr_pixels.size == 0:
        return "unknown"
    mean_bgr = np.mean(bgr_pixels.reshape(-1, 3), axis=0)
    hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue = int(hsv[0])
    sat = int(hsv[1])
    val = int(hsv[2])
    if sat < 60 or val < 80:
        return "unknown"
    if hue < 8 or hue >= 172:
        return "red"
    if 8 <= hue < 24:
        return "orange"
    if 24 <= hue < 42:
        return "yellow"
    if 42 <= hue < 88:
        return "green"
    if 88 <= hue < 105:
        return "cyan"
    if 105 <= hue < 132:
        return "blue"
    if 132 <= hue < 160:
        return "magenta"
    return "pink"


def _find_vertical_split(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad = np.abs(np.diff(gray.astype(np.float32), axis=1))
    column_score = np.mean(grad, axis=0)
    lo = int(w * 0.38)
    hi = int(w * 0.62)
    if hi <= lo:
        return w // 2
    split = int(np.argmax(column_score[lo:hi]) + lo)
    if abs(split - (w // 2)) > int(w * 0.10):
        return w // 2
    return split


def extract_colored_boxes(labeled_image_path: Path) -> Tuple[int, List[BoxLabel]]:
    image = cv2.imread(str(labeled_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load labeled image: {labeled_image_path}")
    h, w = image.shape[:2]
    split_x = _find_vertical_split(image)

    bgr = image.astype(np.int16)
    channel_range = np.max(bgr, axis=2) - np.min(bgr, axis=2)
    bright = np.max(bgr, axis=2)
    color_mask = ((channel_range >= 75) & (bright >= 145)).astype(np.uint8) * 255
    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
    boxes: List[BoxLabel] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 80 or area > 18000:
            continue
        if bw < 18 or bh < 18 or bw > int(w * 0.22) or bh > int(h * 0.22):
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.3:
            continue

        component = labels == label
        color_key = _dominant_color_key(image[component])
        if color_key == "unknown":
            continue

        cx, cy = centroids[label]
        side = "image1" if cx < split_x else "image2"
        boxes.append(
            BoxLabel(
                side=side,
                color_key=color_key,
                center=(float(cx), float(cy)),
                bbox=(x, y, bw, bh),
                area=area,
            )
        )

    return split_x, boxes


def _map_label_point_to_original(
    point: Point,
    side: str,
    split_x: int,
    labeled_size: Tuple[int, int],
    image1: ImageInfo,
    image2: ImageInfo,
) -> Point:
    labeled_w, labeled_h = labeled_size
    if side == "image1":
        local_x = point[0]
        crop_w = split_x
        target = image1
    else:
        local_x = point[0] - split_x
        crop_w = labeled_w - split_x
        target = image2
    x = local_x * target.width / max(1, crop_w)
    y = point[1] * target.height / max(1, labeled_h)
    return float(x), float(y)


def extract_ground_truth_for_folder(folder: Path) -> Dict[str, object]:
    image1, image2, labeled = discover_pair_images(folder)
    split_x, boxes = extract_colored_boxes(labeled.path)
    labeled_size = (labeled.width, labeled.height)

    by_color: Dict[str, Dict[str, BoxLabel]] = {}
    for box in boxes:
        by_color.setdefault(box.color_key, {})
        existing = by_color[box.color_key].get(box.side)
        if existing is None or box.area > existing.area:
            by_color[box.color_key][box.side] = box

    landmarks = []
    for color_key, sides in sorted(by_color.items()):
        if "image1" not in sides or "image2" not in sides:
            continue
        point1 = _map_label_point_to_original(sides["image1"].center, "image1", split_x, labeled_size, image1, image2)
        point2 = _map_label_point_to_original(sides["image2"].center, "image2", split_x, labeled_size, image1, image2)
        landmarks.append(
            {
                "label": color_key,
                "image1_x": round(point1[0], 2),
                "image1_y": round(point1[1], 2),
                "image2_x": round(point2[0], 2),
                "image2_y": round(point2[1], 2),
                "labeled_image1_x": round(sides["image1"].center[0], 2),
                "labeled_image1_y": round(sides["image1"].center[1], 2),
                "labeled_image2_x": round(sides["image2"].center[0], 2),
                "labeled_image2_y": round(sides["image2"].center[1], 2),
            }
        )

    return {
        "pair": folder.name,
        "image1": str(image1.path),
        "image2": str(image2.path),
        "labeled": str(labeled.path),
        "split_x": split_x,
        "landmarks": landmarks,
        "detected_boxes": [
            {
                "side": box.side,
                "color_key": box.color_key,
                "x": round(box.center[0], 2),
                "y": round(box.center[1], 2),
                "bbox": list(box.bbox),
                "area": box.area,
            }
            for box in boxes
        ],
    }


def write_ground_truth(root: Path, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = [extract_ground_truth_for_folder(folder) for folder in discover_pair_folders(root)]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"root": str(root), "pairs": results}, f, indent=2)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair", "label", "image1_x", "image1_y", "image2_x", "image2_y", "image1", "image2", "labeled"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in results:
            for landmark in pair["landmarks"]:
                writer.writerow(
                    {
                        "pair": pair["pair"],
                        "label": landmark["label"],
                        "image1_x": landmark["image1_x"],
                        "image1_y": landmark["image1_y"],
                        "image2_x": landmark["image2_x"],
                        "image2_y": landmark["image2_y"],
                        "image1": pair["image1"],
                        "image2": pair["image2"],
                        "labeled": pair["labeled"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract labeled corresponding landmarks from side-by-side screenshots.")
    parser.add_argument("--root", required=True, type=Path, help="Folder containing Pair 1, Pair 2, etc.")
    parser.add_argument("--output-json", required=True, type=Path, help="Path for extracted ground-truth JSON")
    parser.add_argument("--output-csv", required=True, type=Path, help="Path for extracted ground-truth CSV")
    args = parser.parse_args()
    write_ground_truth(args.root, args.output_json, args.output_csv)
    print(f"Saved ground-truth JSON: {args.output_json}")
    print(f"Saved ground-truth CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

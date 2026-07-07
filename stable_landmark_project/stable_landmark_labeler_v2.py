"""
stable_landmark_labeler_v2.py

Conservative stable skin landmark matcher for longitudinal FLAME/vitiligo imaging.

Example:
    python stable_landmark_labeler_v2.py --image1 images/visit1.jpg --image2 images/visit2.jpg --output-dir outputs --select-polygon-roi --top 6 --lowe-ratio 0.70 --ransac-threshold 4.0 --min-inliers 6 --min-inlier-ratio 0.05 --min-spacing 60 --save-debug

Purpose:
    Match stable surrounding anatomical/skin landmarks between Visit 1 and Visit 2.
    This is not vitiligo detection. The script is intentionally conservative:
    a false correspondence is worse than saying that there are not enough reliable
    landmarks.

Interactive polygon controls:
    Left click: add polygon point
    Enter/Return: finish polygon
    R: reset polygon
    Esc: cancel

Semi-automatic anchor controls:
    Click a trusted landmark in image 1, then click the same landmark in image 2.
    Repeat at least 3 times.
    Enter/Return: finish once at least 3 pairs exist
    U: undo the pending click or last completed pair
    Esc: cancel anchor selection

After the ROI is selected, the script automatically suppresses obvious glare,
very dark background/shadow, low-saturation bright regions, and hair-like line
clutter inside the ROI. RANSAC and the reliability thresholds are still the main
guardrails against false matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
IntPoint = Tuple[int, int]
_HOMOGRAPHY_RANSAC_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


FAIL_MESSAGE = (
    "Not enough reliable corresponding landmarks found. Do not trust this alignment. "
    "Try selecting a tighter overlapping skin ROI, improving lighting, avoiding "
    "ring/glare/hair-heavy areas when drawing the ROI, or using images with more overlap."
)


@dataclass
class MatchQuality:
    total_keypoints_image1: int = 0
    total_keypoints_image2: int = 0
    raw_matches: int = 0
    good_matches_after_lowe: int = 0
    ransac_inliers: int = 0
    inlier_ratio: float = 0.0
    average_reprojection_error: Optional[float] = None
    geometric_model: Optional[str] = None
    sift_contrast_image1: Optional[float] = None
    sift_contrast_image2: Optional[float] = None
    lowe_ratio_used: Optional[float] = None
    spatially_gated_matches: int = 0
    stable_spot_keypoints_image1: int = 0
    stable_spot_keypoints_image2: int = 0
    brown_patch_candidates_image1: int = 0
    brown_patch_candidates_image2: int = 0
    brown_patch_matches: int = 0
    mole_candidates_image1: int = 0
    mole_candidates_image2: int = 0
    mole_constellation_matches: int = 0
    geometry_candidate_matches: int = 0
    geometry_candidate_inliers: int = 0
    anchor_pairs: int = 0
    anchor_average_error: Optional[float] = None
    anchor_model: Optional[str] = None
    use_rootsift: bool = True
    cross_check_matches: bool = True
    homography_method: str = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC"


@dataclass
class LandmarkCandidate:
    match: Optional[cv2.DMatch]
    point1: Point
    point2: Point
    descriptor_distance: float
    reprojection_error: float
    stable_spot_score: float = 0.0
    prominence_score: float = 0.0
    red_brown_signal: float = 0.0
    box_half1: int = 14
    box_half2: int = 14
    source: str = "sift"


@dataclass
class BrownPatchCandidate:
    center: Point
    bbox: Tuple[int, int, int, int]
    area: int
    mean_lab: Tuple[float, float, float]
    score: float
    box_half: int


@dataclass
class MoleCandidate:
    center: Point
    bbox: Tuple[int, int, int, int]
    area: int
    mean_lab: Tuple[float, float, float]
    score: float
    prominence: float
    box_half: int


@dataclass
class StableCandidate:
    center: Point
    area: int
    mean_lab: Tuple[float, float, float]
    score: float
    prominence: float
    box_half: int
    source: str


@dataclass
class AnchorPair:
    point1: Point
    point2: Point


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def load_mask(path: Path, image_shape: Tuple[int, int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and enhance local contrast for low-texture skin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def to_rootsift(descriptors: Optional[np.ndarray], eps: float = 1e-7) -> Optional[np.ndarray]:
    if descriptors is None or len(descriptors) == 0:
        return descriptors
    descriptors = descriptors.astype(np.float32)
    l1 = np.sum(np.abs(descriptors), axis=1, keepdims=True)
    descriptors = descriptors / (l1 + eps)
    return np.sqrt(descriptors)


def _resize_for_display(image: np.ndarray, display_scale: float) -> np.ndarray:
    if display_scale <= 0:
        raise ValueError("--display-scale must be > 0")
    if display_scale == 1.0:
        return image.copy()
    return cv2.resize(image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)


def _draw_polygon_preview(
    display: np.ndarray,
    points: Sequence[IntPoint],
    title: str,
    committed_polygons: Optional[Sequence[Sequence[IntPoint]]] = None,
) -> np.ndarray:
    preview = display.copy()
    if committed_polygons:
        for poly in committed_polygons:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                overlay = preview.copy()
                cv2.fillPoly(overlay, [pts], (0, 140, 255))
                preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
                cv2.polylines(preview, [pts], True, (0, 140, 255), 2)

    for pt in points:
        cv2.circle(preview, pt, 4, (0, 255, 255), -1)
    if len(points) >= 2:
        cv2.polylines(preview, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)

    cv2.putText(preview, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(preview, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return preview


def select_polygon_mask(image: np.ndarray, title: str, display_scale: float = 0.75) -> Optional[np.ndarray]:
    display = _resize_for_display(image, display_scale)
    window_name = title
    points: List[IntPoint] = []
    finished = False
    cancelled = False

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        prompt = "Left click points | Enter finish | R reset | Esc cancel"
        cv2.imshow(window_name, _draw_polygon_preview(display, points, prompt))
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            if len(points) >= 3:
                finished = True
                break
            print(f"{title}: need at least 3 polygon points.")
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key == 27:
            cancelled = True
            break

    cv2.destroyWindow(window_name)

    if cancelled or not finished:
        return None

    original_points = np.array(
        [(round(x / display_scale), round(y / display_scale)) for x, y in points],
        dtype=np.int32,
    )
    h, w = image.shape[:2]
    original_points[:, 0] = np.clip(original_points[:, 0], 0, w - 1)
    original_points[:, 1] = np.clip(original_points[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [original_points], 255)
    return mask


def _anchor_to_candidate(anchor: AnchorPair, index: int) -> LandmarkCandidate:
    return LandmarkCandidate(
        match=None,
        point1=anchor.point1,
        point2=anchor.point2,
        descriptor_distance=0.0,
        reprojection_error=0.0,
        stable_spot_score=999.0,
        prominence_score=100000.0 - index,
        red_brown_signal=999.0,
        box_half1=18,
        box_half2=18,
        source="anchor",
    )


def load_anchor_pairs(path: Path) -> List[AnchorPair]:
    if not path.exists():
        raise FileNotFoundError(f"Anchor file does not exist: {path}")

    anchors: List[AnchorPair] = []
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                anchors.append(
                    AnchorPair(
                        (float(row["image1_x"]), float(row["image1_y"])),
                        (float(row["image2_x"]), float(row["image2_y"])),
                    )
                )
        return anchors

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("anchors", data if isinstance(data, list) else [])
    for item in records:
        anchors.append(
            AnchorPair(
                (float(item["image1_x"]), float(item["image1_y"])),
                (float(item["image2_x"]), float(item["image2_y"])),
            )
        )
    return anchors


def save_anchor_pairs(path: Path, anchors: Sequence[AnchorPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors": [
            {
                "anchor_id": index,
                "image1_x": round(float(anchor.point1[0]), 2),
                "image1_y": round(float(anchor.point1[1]), 2),
                "image2_x": round(float(anchor.point2[0]), 2),
                "image2_y": round(float(anchor.point2[1]), 2),
            }
            for index, anchor in enumerate(anchors, start=1)
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draw_anchor_preview(
    combined: np.ndarray,
    split_x: int,
    anchors: Sequence[Tuple[IntPoint, IntPoint]],
    pending: Optional[IntPoint],
    prompt: str,
) -> np.ndarray:
    preview = combined.copy()
    cv2.line(preview, (split_x, 0), (split_x, preview.shape[0] - 1), (20, 20, 20), 2)
    palette = [
        (0, 0, 255),
        (0, 128, 255),
        (0, 210, 255),
        (0, 190, 0),
        (255, 0, 0),
        (255, 0, 255),
        (170, 0, 255),
        (0, 255, 255),
    ]
    for index, (p1, p2) in enumerate(anchors, start=1):
        color = palette[(index - 1) % len(palette)]
        cv2.rectangle(preview, (p1[0] - 12, p1[1] - 12), (p1[0] + 12, p1[1] + 12), color, 2)
        cv2.rectangle(preview, (p2[0] - 12, p2[1] - 12), (p2[0] + 12, p2[1] + 12), color, 2)
        _label_text(preview, str(index), (p1[0] + 15, max(24, p1[1] - 15)), 0.65, color)
        _label_text(preview, str(index), (p2[0] + 15, max(24, p2[1] - 15)), 0.65, color)
    if pending is not None:
        cv2.drawMarker(preview, pending, (255, 255, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    _label_text(preview, prompt, (16, 34), 0.72, (255, 255, 255))
    return preview


def select_anchor_pairs_interactive(
    img1: np.ndarray,
    img2: np.ndarray,
    requested_count: int,
    display_scale: float,
) -> List[AnchorPair]:
    draw1 = _resize_for_display(img1, display_scale)
    draw2 = _resize_for_display(img2, display_scale)
    draw1, draw2 = _pad_to_same_height(draw1, draw2)
    split_x = draw1.shape[1]
    combined = np.hstack([draw1, draw2])
    window_name = "Select trusted corresponding anchors"
    anchors_display: List[Tuple[IntPoint, IntPoint]] = []
    pending: Optional[IntPoint] = None
    pending_side: Optional[str] = None

    def on_mouse(event, x, y, _flags, _param):
        nonlocal pending, pending_side
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        side = "image1" if x < split_x else "image2"
        if pending is None:
            if side != "image1":
                print("Click the anchor in image 1 first, then its matching point in image 2.")
                return
            pending = (x, y)
            pending_side = side
            return
        if pending_side == side:
            print("Now click the corresponding anchor in the other image.")
            return
        anchors_display.append((pending, (x, y)))
        pending = None
        pending_side = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        needed = max(0, requested_count - len(anchors_display))
        prompt = (
            f"Click image1 anchor then matching image2 anchor | anchors={len(anchors_display)}"
            f"{'' if requested_count <= 0 else f'/{requested_count}'} | Enter finish | U undo | Esc cancel"
        )
        cv2.imshow(window_name, _draw_anchor_preview(combined, split_x, anchors_display, pending, prompt))
        key = cv2.waitKey(20) & 0xFF
        if requested_count > 0 and needed == 0:
            break
        if key in (13, 10):
            if len(anchors_display) >= 3:
                break
            print("Need at least 3 anchor pairs for anchor mode.")
        elif key in (ord("u"), ord("U")):
            if pending is not None:
                pending = None
                pending_side = None
            elif anchors_display:
                anchors_display.pop()
        elif key == 27:
            anchors_display = []
            break

    cv2.destroyWindow(window_name)

    anchors: List[AnchorPair] = []
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    for p1, p2 in anchors_display:
        x1 = float(np.clip(p1[0] / display_scale, 0, w1 - 1))
        y1 = float(np.clip(p1[1] / display_scale, 0, h1 - 1))
        x2 = float(np.clip((p2[0] - split_x) / display_scale, 0, w2 - 1))
        y2 = float(np.clip(p2[1] / display_scale, 0, h2 - 1))
        anchors.append(AnchorPair((x1, y1), (x2, y2)))
    return anchors


def _roi_values(channel: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    values = channel[roi_mask > 0]
    if values.size == 0:
        return channel.reshape(-1)
    return values


def _remove_hair_like_components(mask: np.ndarray) -> np.ndarray:
    """Keep elongated dark-line components while avoiding small round moles."""
    component_mask = np.zeros_like(mask)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        longest = max(w, h)
        shortest = max(1, min(w, h))
        aspect_ratio = longest / shortest

        if area >= 12 and longest >= 10 and aspect_ratio >= 3.0:
            component_mask[labels == label] = 255

    return component_mask


def build_skin_color_mask(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Approximate skin-colored regions so patterned clothing is not treated as ROI.

    This is deliberately permissive for clinical photos: it keeps normal skin,
    mildly red skin, and pale/depigmented skin, while rejecting very neutral
    white/black fabric and highly patterned non-skin regions.
    """
    bgr = image.astype(np.float32)
    b, g, r = cv2.split(bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    color_range = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    red_over_blue = r - b
    warm_skin = (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 70)
        & (cb <= 145)
        & (red_over_blue >= 4)
        & (value > 28)
    )
    pale_skin = (
        (cr >= 124)
        & (cr <= 158)
        & (cb >= 84)
        & (cb <= 142)
        & (a_chan >= 118)
        & (b_chan >= 120)
        & (red_over_blue >= -4)
        & (color_range >= 7)
        & (value > 42)
    )
    red_skin = (
        (cr >= 145)
        & (cb <= 150)
        & (r >= g * 0.95)
        & (value > 30)
    )

    skin = ((warm_skin | pale_skin | red_skin) & (roi_mask > 0)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if num_labels <= 1:
        return roi_mask.copy()

    min_area = max(250, int(0.004 * image.shape[0] * image.shape[1]))
    filtered = np.zeros_like(skin)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = w / max(1, h)
        thin_horizontal = aspect > 3.2 and h < int(0.16 * image.shape[0])
        touches_top = y < int(0.04 * image.shape[0])
        if area >= min_area and not (thin_horizontal and touches_top):
            filtered[labels == label] = 255

    if np.count_nonzero(filtered) < 0.08 * max(1, np.count_nonzero(roi_mask)):
        return roi_mask.copy()
    return filtered


def build_automatic_quality_mask(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Build a conservative no-click cleanup mask inside the skin ROI.

    This is intentionally simple OpenCV logic, not lesion detection:
    - suppress extremely bright/low-saturation glare or metal-like areas
    - suppress very dark background/shadow regions
    - suppress elongated dark hair-like lines while preserving small round spots
    - erode the ROI edge slightly so labels are less likely to land on borders
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    roi_sat = _roi_values(saturation, roi_mask)
    roi_value = _roi_values(value, roi_mask)
    skin_roi_mask = build_skin_color_mask(image, roi_mask)

    dark_threshold = max(12, int(np.percentile(roi_value, 2)))
    bright_threshold = min(245, int(np.percentile(roi_value, 98)))
    low_sat_threshold = max(28, int(np.percentile(roi_sat, 20)))

    dark_mask = ((value <= dark_threshold) & (skin_roi_mask > 0)).astype(np.uint8) * 255
    glare_mask = ((value >= bright_threshold) & (saturation <= low_sat_threshold) & (skin_roi_mask > 0)).astype(np.uint8) * 255

    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel9)
    hair_threshold = max(18, int(np.percentile(_roi_values(blackhat, roi_mask), 97)))
    hair_candidates = ((blackhat >= hair_threshold) & (skin_roi_mask > 0)).astype(np.uint8) * 255
    hair_mask = _remove_hair_like_components(hair_candidates)

    suppress = cv2.bitwise_or(dark_mask, glare_mask)
    suppress = cv2.bitwise_or(suppress, hair_mask)
    suppress = cv2.dilate(suppress, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    edge_safe_roi = cv2.erode(skin_roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    allowed = cv2.bitwise_and(edge_safe_roi, cv2.bitwise_not(suppress))
    allowed = cv2.morphologyEx(allowed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return allowed


def build_allowed_mask(
    image: np.ndarray,
    use_polygon_roi: bool,
    title: str,
    display_scale: float,
    use_auto_quality_mask: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    if use_polygon_roi:
        roi_mask = select_polygon_mask(image, f"{title} skin ROI", display_scale)
        if roi_mask is None:
            print(f"{title}: no polygon ROI selected; using full image.")
            roi_mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        roi_mask = np.full((h, w), 255, dtype=np.uint8)

    if use_auto_quality_mask:
        allowed = build_automatic_quality_mask(image, roi_mask)
        kept = int(np.count_nonzero(allowed))
        total = int(np.count_nonzero(roi_mask))
        kept_percent = 100.0 * kept / max(total, 1)
        print(f"{title}: automatic quality mask kept {kept_percent:.1f}% of the ROI.")
    else:
        allowed = roi_mask.copy()

    return allowed


def suppress_ring_annulus_from_mask(
    mask: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
    inner_radius: Optional[float] = None,
    padding: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Remove the metal ring band from detection while keeping skin inside the ring.

    The older ring suppression removed one filled circle. That was too aggressive
    because it erased valid skin landmarks inside or near the ring. This keeps the
    center region available and suppresses only an annulus around the visible band.
    """
    output = mask.copy()
    cx, cy = int(round(center[0])), int(round(center[1]))
    outer = max(1, int(round(outer_radius + padding)))
    if inner_radius is None:
        inner_radius = outer_radius * 0.52
    inner = max(1, int(round(inner_radius - padding)))
    inner = min(inner, max(1, outer - 2))

    ring_band = np.zeros_like(mask)
    cv2.circle(ring_band, (cx, cy), outer, 255, -1)
    cv2.circle(ring_band, (cx, cy), inner, 0, -1)
    output[ring_band > 0] = 0
    return output, float(inner)


def _ring_annulus_mask(shape: Tuple[int, int], center: Tuple[float, float], outer_radius: float, inner_radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(mask, (cx, cy), int(round(outer_radius)), 255, -1)
    cv2.circle(mask, (cx, cy), int(round(inner_radius)), 0, -1)
    return mask


def _ring_material_score(image: np.ndarray, center: Point, radius: float) -> float:
    """Estimate whether a detected circle is actual non-skin ring material."""
    h, w = image.shape[:2]
    outer = min(radius + 8.0, max(h, w))
    inner = max(2.0, radius * 0.52 - 8.0)
    annulus = _ring_annulus_mask((h, w), center, outer, inner)
    values = annulus > 0
    if np.count_nonzero(values) < 50:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    skin = build_skin_color_mask(image, np.full((h, w), 255, dtype=np.uint8))
    center_disk = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        center_disk,
        (int(round(center[0])), int(round(center[1]))),
        max(2, int(round(radius * 0.40))),
        255,
        -1,
    )
    center_values = center_disk > 0
    if np.count_nonzero(center_values) < 25:
        return 0.0
    center_skin_fraction = float(np.mean(skin[center_values] > 0))
    if center_skin_fraction < 0.35:
        return 0.0

    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)

    low_skin_fraction = float(np.mean(skin[values] == 0))
    if low_skin_fraction < 0.60:
        return 0.0
    metal_like_fraction = float(np.mean((sat[values] < 55) & (value[values] > 105)))
    edge_strength = min(1.0, float(np.percentile(gradient[values], 75)) / 90.0)
    return (0.50 * low_skin_fraction) + (0.30 * metal_like_fraction) + (0.20 * edge_strength)


def detect_largest_ring_circle(image: np.ndarray) -> Optional[Tuple[Point, float]]:
    """Find a likely circular metal measurement ring in a baseline image."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_dim = min(h, w)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(80, int(min_dim * 0.28)),
        param1=90,
        param2=32,
        minRadius=max(20, int(min_dim * 0.10)),
        maxRadius=max(25, int(min_dim * 0.34)),
    )
    if circles is None:
        return None

    candidates = np.round(circles[0]).astype(int)
    scored = []
    for cx, cy, radius in candidates:
        border_slack = max(8.0, 0.12 * float(radius))
        if (
            cx - radius < -border_slack
            or cy - radius < -border_slack
            or cx + radius > w + border_slack
            or cy + radius > h + border_slack
        ):
            continue
        score = _ring_material_score(image, (float(cx), float(cy)), float(radius))
        if score >= 0.42:
            scored.append((score, int(radius), float(cx), float(cy)))
    if not scored:
        return None

    _score, radius, cx, cy = max(scored, key=lambda item: (item[0], item[1]))
    return (float(cx), float(cy)), float(radius)


def detect_sift_features(
    gray: np.ndarray,
    mask: Optional[np.ndarray],
    max_features: int = 8000,
    contrast_threshold: float = 0.01,
    use_rootsift: bool = True,
):
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if use_rootsift:
        descriptors = to_rootsift(descriptors)
    return keypoints, descriptors


def detect_sift_features_adaptive(
    gray: np.ndarray,
    mask: Optional[np.ndarray],
    max_features: int,
    contrast_threshold: float,
    target_keypoints: int,
    use_rootsift: bool = True,
):
    """
    Smooth skin can produce very few SIFT features at a fixed threshold.
    Try progressively lower thresholds, keeping the first result that gives
    enough candidate points or the richest result available.
    """
    thresholds = []
    for threshold in [contrast_threshold, 0.006, 0.003, 0.0015, 0.0008]:
        if threshold > 0 and threshold not in thresholds:
            thresholds.append(threshold)

    best_keypoints = []
    best_descriptors = None
    best_threshold = thresholds[0]

    for threshold in thresholds:
        keypoints, descriptors = detect_sift_features(gray, mask, max_features, threshold, use_rootsift)
        if len(keypoints) > len(best_keypoints):
            best_keypoints = keypoints
            best_descriptors = descriptors
            best_threshold = threshold
        if len(keypoints) >= target_keypoints:
            return keypoints, descriptors, threshold

    return best_keypoints, best_descriptors, best_threshold


def match_descriptors(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    lowe_ratio: float,
    cross_check: bool = True,
):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], []

    des1 = np.asarray(des1, dtype=np.float32)
    des2 = np.asarray(des2, dtype=np.float32)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    if cross_check and good:
        reverse = flann.knnMatch(des2, des1, k=1)
        best_back = {}
        for pair in reverse:
            if pair:
                rm = pair[0]
                best_back[rm.queryIdx] = rm.trainIdx
        good = [m for m in good if best_back.get(m.trainIdx) == m.queryIdx]
    return raw_matches, good


def compute_mask_prior_affine(mask1: np.ndarray, mask2: np.ndarray) -> Optional[np.ndarray]:
    pts1_yx = np.column_stack(np.where(mask1 > 0))
    pts2_yx = np.column_stack(np.where(mask2 > 0))
    if len(pts1_yx) < 20 or len(pts2_yx) < 20:
        return None

    pts1 = pts1_yx[:, ::-1].astype(np.float32)
    pts2 = pts2_yx[:, ::-1].astype(np.float32)

    def pca_basis(points: np.ndarray):
        center = points.mean(axis=0)
        centered = points - center
        covariance = np.cov(centered.T)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]

        # Make the long axis point generally downward in image coordinates.
        if vectors[1, 0] < 0:
            vectors[:, 0] *= -1
        if np.linalg.det(vectors) < 0:
            vectors[:, 1] *= -1

        scales = np.sqrt(np.maximum(values, 1e-6))
        return center, vectors, scales

    center1, basis1, scale1 = pca_basis(pts1)
    center2, basis2, scale2 = pca_basis(pts2)
    linear = basis2 @ np.diag(scale2 / scale1) @ basis1.T
    translation = center2 - linear @ center1
    return np.vstack([np.column_stack([linear, translation]), [0.0, 0.0, 1.0]])


def filter_matches_by_spatial_prior(kp1, kp2, matches, prior_matrix: Optional[np.ndarray], max_distance: float):
    if prior_matrix is None:
        return list(matches)

    filtered = []
    best_by_train: Dict[int, cv2.DMatch] = {}
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        predicted = prior_matrix @ np.array([x1, y1, 1.0], dtype=np.float64)
        x2, y2 = kp2[match.trainIdx].pt
        if math.hypot(float(predicted[0]) - x2, float(predicted[1]) - y2) > max_distance:
            continue

        current = best_by_train.get(match.trainIdx)
        if current is None or match.distance < current.distance:
            best_by_train[match.trainIdx] = match

    filtered = list(best_by_train.values())
    filtered.sort(key=lambda m: m.distance)
    return filtered


def match_descriptors_adaptive(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    kp1,
    kp2,
    lowe_ratio: float,
    spatial_prior: Optional[np.ndarray],
    spatial_gate: float,
    target_matches: int,
    cross_check: bool = True,
):
    ratios = []
    for ratio in [lowe_ratio, 0.75, 0.80, 0.85, 0.90, 0.95]:
        if lowe_ratio <= ratio <= 0.95 and ratio not in ratios:
            ratios.append(ratio)

    best_raw = []
    best_matches = []
    best_ratio = lowe_ratio

    for ratio in ratios:
        raw, good = match_descriptors(des1, des2, ratio, cross_check)
        gated = filter_matches_by_spatial_prior(kp1, kp2, good, spatial_prior, spatial_gate)
        if len(gated) > len(best_matches):
            best_raw = raw
            best_matches = gated
            best_ratio = ratio
        if len(gated) >= target_matches:
            return raw, gated, ratio

    return best_raw, best_matches, best_ratio


def compute_homography_ransac(kp1, kp2, matches, threshold: float):
    if len(matches) < 4:
        return None, [], None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    try:
        H, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            _HOMOGRAPHY_RANSAC_METHOD,
            threshold,
            maxIters=10000,
            confidence=0.9999,
        )
    except cv2.error:
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

    if H is None or inlier_mask is None:
        return None, [], None

    inlier_flags = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, inlier_flags) if keep]
    return H, inlier_matches, inlier_flags


def compute_affine_ransac(kp1, kp2, matches, threshold: float):
    if len(matches) < 3:
        return None, [], None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )

    if affine is None or inlier_mask is None:
        return None, [], None

    matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
    inlier_flags = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, inlier_flags) if keep]
    return matrix, inlier_matches, inlier_flags


def compute_geometric_model_ransac(kp1, kp2, matches, threshold: float):
    homography, homography_inliers, homography_flags = compute_homography_ransac(kp1, kp2, matches, threshold)
    affine, affine_inliers, affine_flags = compute_affine_ransac(kp1, kp2, matches, threshold)

    candidates = []
    if homography is not None:
        candidates.append(("homography", homography, homography_inliers, homography_flags))
    if affine is not None:
        candidates.append(("affine", affine, affine_inliers, affine_flags))

    if not candidates:
        return None, [], None, None

    def rank(candidate):
        model_name, matrix, inlier_matches, _flags = candidate
        errors = compute_reprojection_errors(matrix, kp1, kp2, inlier_matches)
        avg_error = float(np.mean(errors)) if errors else float("inf")
        model_bonus = 0.25 if model_name == "homography" else 0.0
        return (len(inlier_matches), -avg_error, model_bonus)

    best_model_name, best_matrix, best_inliers, best_flags = max(candidates, key=rank)
    return best_matrix, best_inliers, best_flags, best_model_name


def compute_reprojection_errors(matrix: np.ndarray, kp1, kp2, matches: Sequence[cv2.DMatch]) -> List[float]:
    if matrix is None or len(matches) == 0:
        return []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src_pts, matrix).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    return [float(err) for err in errors]


def compute_anchor_geometry(anchors: Sequence[AnchorPair]) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    if len(anchors) < 3:
        return None, None, None

    src = np.float32([anchor.point1 for anchor in anchors])
    dst = np.float32([anchor.point2 for anchor in anchors])
    candidates: List[Tuple[str, np.ndarray, float]] = []

    if len(anchors) == 3:
        affine_2x3 = cv2.getAffineTransform(src[:3], dst[:3])
    else:
        affine_2x3, _inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.LMEDS,
            refineIters=20,
        )
    if affine_2x3 is not None:
        affine = np.vstack([affine_2x3, [0.0, 0.0, 1.0]]).astype(np.float32)
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), affine).reshape(-1, 2)
        avg_error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
        candidates.append(("anchor_affine", affine, avg_error))

    if len(anchors) >= 4:
        homography, _mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), 0)
        if homography is not None:
            projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
            avg_error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
            candidates.append(("anchor_homography", homography.astype(np.float32), avg_error))

    if not candidates:
        return None, None, None

    model_name, matrix, avg_error = min(candidates, key=lambda item: (item[2], 0 if item[0] == "anchor_affine" else 1))
    return matrix, model_name, avg_error


def _far_enough(point: Point, selected_points: Iterable[Point], min_spacing: float) -> bool:
    return all(math.hypot(point[0] - prev[0], point[1] - prev[1]) >= min_spacing for prev in selected_points)


def _patch_bounds(image_shape: Tuple[int, int, int], point: Point, radius: int):
    h, w = image_shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    return x1, y1, x2, y2




def stable_spot_measurement(
    image: np.ndarray,
    point: Point,
    inner_radius: int = 4,
    outer_radius: int = 18,
) -> Dict[str, float]:
    """
    Measure whether a point looks like a visible stable dark/red-brown skin mark.

    Higher scores favor prominent moles/freckles/dark spots. Low-saturation gray
    areas are penalized because they are often metal, glare, or non-skin.
    """
    h, w = image.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    if x < 0 or y < 0 or x >= w or y >= h:
        return {
            "score": 0.0,
            "prominence": 0.0,
            "red_brown_signal": 0.0,
            "box_half": 14.0,
            "skin_penalty": 100.0,
        }

    x1, y1, x2, y2 = _patch_bounds(image.shape, point, outer_radius)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return {
            "score": 0.0,
            "prominence": 0.0,
            "red_brown_signal": 0.0,
            "box_half": 14.0,
            "skin_penalty": 100.0,
        }

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inner = distance <= inner_radius
    annulus = (distance > inner_radius + 2) & (distance <= outer_radius)
    if not np.any(inner) or not np.any(annulus):
        return {
            "score": 0.0,
            "prominence": 0.0,
            "red_brown_signal": 0.0,
            "box_half": 14.0,
            "skin_penalty": 100.0,
        }

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    bgr = patch.astype(np.float32)

    inner_l = float(np.mean(lab[:, :, 0][inner]))
    outer_l = float(np.mean(lab[:, :, 0][annulus]))
    inner_sat = float(np.mean(hsv[:, :, 1][inner]))
    inner_value = float(np.mean(hsv[:, :, 2][inner]))
    outer_value = float(np.mean(hsv[:, :, 2][annulus]))
    outer_sat = float(np.mean(hsv[:, :, 1][annulus]))

    inner_b = float(np.mean(bgr[:, :, 0][inner]))
    inner_g = float(np.mean(bgr[:, :, 1][inner]))
    inner_r = float(np.mean(bgr[:, :, 2][inner]))
    outer_b = float(np.mean(bgr[:, :, 0][annulus]))
    outer_g = float(np.mean(bgr[:, :, 1][annulus]))
    outer_r = float(np.mean(bgr[:, :, 2][annulus]))

    dark_contrast = max(0.0, outer_l - inner_l)
    value_contrast = max(0.0, outer_value - inner_value)
    red_brown_center = max(0.0, inner_r - (0.55 * inner_g + 0.45 * inner_b))
    red_brown_contrast = max(0.0, (inner_r - inner_g) - (outer_r - outer_g))
    red_brown_signal = red_brown_center + red_brown_contrast
    visible_contrast = max(dark_contrast, value_contrast, red_brown_contrast)

    bright_patch_penalty = max(0.0, inner_value - outer_value) * 0.45
    low_sat_bright_penalty = 18.0 if inner_sat < 28 and inner_value > outer_value else 0.0
    gray_non_skin_penalty = 35.0 if inner_sat < 18 and outer_sat < 28 else 0.0
    weak_visibility_penalty = max(0.0, 18.0 - visible_contrast) * 1.4

    score = (
        (1.15 * dark_contrast)
        + (0.60 * value_contrast)
        + (0.95 * red_brown_center)
        + (0.95 * red_brown_contrast)
        - bright_patch_penalty
        - low_sat_bright_penalty
        - gray_non_skin_penalty
        - weak_visibility_penalty
    )

    local_bg = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 4)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark_map = np.maximum(0.0, local_bg - gray)
    red_map = np.maximum(0.0, bgr[:, :, 2] - (0.55 * bgr[:, :, 1] + 0.45 * bgr[:, :, 0]))
    mark_map = ((dark_map + 0.5 * red_map) >= max(8.0, 0.45 * visible_contrast)).astype(np.uint8)
    mark_map[distance > outer_radius] = 0

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mark_map, connectivity=8)
    box_half = 14.0
    if num_labels > 1 and 0 <= y - y1 < labels.shape[0] and 0 <= x - x1 < labels.shape[1]:
        label = labels[y - y1, x - x1]
        if label > 0:
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            box_half = float(np.clip(max(width, height) * 0.75 + 8.0, 10.0, 34.0))

    prominence = max(0.0, score) + (0.45 * visible_contrast) + (0.20 * red_brown_center)
    skin_penalty = gray_non_skin_penalty + low_sat_bright_penalty
    return {
        "score": float(max(0.0, score)),
        "prominence": float(max(0.0, prominence)),
        "red_brown_signal": float(max(0.0, red_brown_signal)),
        "box_half": box_half,
        "skin_penalty": float(skin_penalty),
    }


def stable_spot_score(image: np.ndarray, point: Point, inner_radius: int = 4, outer_radius: int = 18) -> float:
    return stable_spot_measurement(image, point, inner_radius, outer_radius)["score"]


def detect_brown_patch_candidates(
    image: np.ndarray,
    mask: np.ndarray,
    max_patches: int,
    min_area: int,
    max_area: int,
    min_score: float,
) -> List[BrownPatchCandidate]:
    """
    Detect medium brown/red-brown patch landmarks that may not produce strong
    point keypoints. These are candidate stable pigmented regions, not vitiligo
    segmentation.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    bgr = image.astype(np.float32)
    b, g, r = cv2.split(bgr)

    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if np.count_nonzero(mask) == 0:
        return []

    local_l = cv2.GaussianBlur(l_chan, (0, 0), 15)
    dark_contrast = np.maximum(0.0, local_l - l_chan)
    red_brown = np.maximum(0.0, r - (0.54 * g + 0.46 * b))
    lab_warmth = np.maximum(0.0, (a_chan - 128.0) * 0.7 + (b_chan - 128.0) * 0.3)
    brown_score = (0.95 * dark_contrast) + (0.75 * red_brown) + (0.45 * lab_warmth)

    roi_scores = brown_score[mask > 0]
    if roi_scores.size == 0:
        return []

    # Use a high percentile so broad mild pigmentation is considered, but the
    # whole arm does not become one candidate.
    threshold = max(float(np.percentile(roi_scores, 96.5)), min_score)
    candidate_mask = (
        (brown_score >= threshold)
        & (mask > 0)
        & (sat > 18)
        & (value > 25)
    ).astype(np.uint8) * 255

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel3)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel7)

    patches: List[BrownPatchCandidate] = []

    def add_components(
        component_mask: np.ndarray,
        area_floor: int,
        area_ceiling: int,
        score_floor: float,
        max_width: int,
        box_scale: float,
        duplicate_distance: float,
    ) -> None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < area_floor or area > area_ceiling:
                continue
            if max(w, h) > max_width or min(w, h) < 3:
                continue

            component = labels == label
            mean_score = float(np.mean(brown_score[component]))
            if mean_score < score_floor:
                continue

            cx, cy = centroids[label]
            if any(math.hypot(cx - patch.center[0], cy - patch.center[1]) < duplicate_distance for patch in patches):
                continue

            mean_lab = tuple(float(v) for v in np.mean(lab[component], axis=0))
            box_half = int(np.clip(max(w, h) * box_scale + 8, 12, 110))
            patches.append(
                BrownPatchCandidate(
                    center=(float(cx), float(cy)),
                    bbox=(x, y, w, h),
                    area=area,
                    mean_lab=mean_lab,
                    score=mean_score,
                    box_half=box_half,
                )
            )

    add_components(
        candidate_mask,
        min_area,
        max_area,
        min_score,
        180,
        0.62,
        12.0,
    )

    # A second pass catches larger, softer brown patches that are visually useful
    # but may not cross the compact-spot threshold everywhere.
    soft_score = cv2.GaussianBlur(brown_score, (0, 0), 4)
    soft_roi_scores = soft_score[mask > 0]
    soft_threshold = max(float(np.percentile(soft_roi_scores, 93.0)), min_score * 0.72)
    soft_mask = (
        (soft_score >= soft_threshold)
        & (mask > 0)
        & (sat > 10)
        & (value > 25)
    ).astype(np.uint8) * 255
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    soft_mask = cv2.morphologyEx(soft_mask, cv2.MORPH_OPEN, kernel5)
    soft_mask = cv2.morphologyEx(soft_mask, cv2.MORPH_CLOSE, kernel13)
    add_components(
        soft_mask,
        max(min_area * 3, 55),
        max_area * 2,
        min_score * 0.72,
        240,
        0.58,
        28.0,
    )

    patches.sort(key=lambda patch: patch.score * math.sqrt(max(1, patch.area)), reverse=True)
    return patches[:max_patches]


def match_brown_patch_candidates(
    patches1: Sequence[BrownPatchCandidate],
    patches2: Sequence[BrownPatchCandidate],
    geometry: Optional[np.ndarray],
    max_projected_distance: float,
    max_lab_distance: float,
    max_area_ratio: float,
) -> List[LandmarkCandidate]:
    if geometry is None or not patches1 or not patches2:
        return []

    possible = []
    for i, patch1 in enumerate(patches1):
        p1 = np.float32([[patch1.center]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(p1, geometry).reshape(2)
        for j, patch2 in enumerate(patches2):
            spatial_error = math.hypot(projected[0] - patch2.center[0], projected[1] - patch2.center[1])
            if spatial_error > max_projected_distance:
                continue

            lab_distance = float(np.linalg.norm(np.array(patch1.mean_lab) - np.array(patch2.mean_lab)))
            if lab_distance > max_lab_distance:
                continue

            area_ratio = max(patch1.area, patch2.area) / max(1, min(patch1.area, patch2.area))
            if area_ratio > max_area_ratio:
                continue

            score = (
                (spatial_error / max(max_projected_distance, 1e-6))
                + (0.45 * lab_distance / max(max_lab_distance, 1e-6))
                + (0.25 * math.log(area_ratio))
                - (0.012 * min(patch1.score, patch2.score))
            )
            possible.append((score, i, j, spatial_error, lab_distance))

    selected = []
    used1 = set()
    used2 = set()
    for _score, i, j, spatial_error, lab_distance in sorted(possible, key=lambda item: item[0]):
        if i in used1 or j in used2:
            continue
        patch1 = patches1[i]
        patch2 = patches2[j]
        used1.add(i)
        used2.add(j)
        selected.append(
            LandmarkCandidate(
                match=None,
                point1=patch1.center,
                point2=patch2.center,
                descriptor_distance=lab_distance,
                reprojection_error=float(spatial_error),
                stable_spot_score=min(patch1.score, patch2.score),
                prominence_score=min(
                    patch1.score * math.sqrt(max(1, patch1.area)),
                    patch2.score * math.sqrt(max(1, patch2.area)),
                ),
                red_brown_signal=min(patch1.score, patch2.score),
                box_half1=patch1.box_half,
                box_half2=patch2.box_half,
                source="brown_patch",
            )
        )

    return selected


def _mask_bounds(mask: np.ndarray) -> Tuple[float, float, float, float]:
    points = cv2.findNonZero(mask)
    if points is None:
        h, w = mask.shape[:2]
        return 0.0, 0.0, float(w), float(h)
    x, y, w, h = cv2.boundingRect(points)
    return float(x), float(y), float(max(w, 1)), float(max(h, 1))


def _normalized_point(point: Point, bounds: Tuple[float, float, float, float]) -> Point:
    x, y, w, h = bounds
    return ((point[0] - x) / w, (point[1] - y) / h)


def match_prominent_patches_by_position(
    patches1: Sequence[BrownPatchCandidate],
    patches2: Sequence[BrownPatchCandidate],
    mask1: np.ndarray,
    mask2: np.ndarray,
    existing_matches: Sequence[LandmarkCandidate],
    max_matches: int = 3,
    max_normalized_distance: float = 0.18,
    min_score: float = 32.0,
    max_area_ratio: float = 7.0,
    max_box_half: int = 62,
) -> List[LandmarkCandidate]:
    """
    Fallback for prominent stable patches far from the main RANSAC cluster.

    Homography can extrapolate poorly when all confident matches sit in one part
    of the image. This pass only considers visually prominent patch candidates
    and matches them by normalized ROI position plus size, so broad obvious marks
    are not lost solely because the global model bends too much at the edge.
    """
    if not patches1 or not patches2:
        return []

    used1 = {tuple(candidate.point1) for candidate in existing_matches}
    used2 = {tuple(candidate.point2) for candidate in existing_matches}
    bounds1 = _mask_bounds(mask1)
    bounds2 = _mask_bounds(mask2)

    possible = []
    for i, patch1 in enumerate(patches1):
        if patch1.score < min_score or tuple(patch1.center) in used1:
            continue
        if patch1.box_half > max_box_half:
            continue
        n1 = _normalized_point(patch1.center, bounds1)
        prominence1 = patch1.score * math.sqrt(max(1, patch1.area))
        for j, patch2 in enumerate(patches2):
            if patch2.score < min_score or tuple(patch2.center) in used2:
                continue
            if patch2.box_half > max_box_half:
                continue
            area_ratio = max(patch1.area, patch2.area) / max(1, min(patch1.area, patch2.area))
            if area_ratio > max_area_ratio:
                continue

            n2 = _normalized_point(patch2.center, bounds2)
            norm_distance = math.hypot(n1[0] - n2[0], n1[1] - n2[1])
            if norm_distance > max_normalized_distance:
                continue

            lab_distance = float(np.linalg.norm(np.array(patch1.mean_lab) - np.array(patch2.mean_lab)))
            prominence2 = patch2.score * math.sqrt(max(1, patch2.area))
            prominence = min(prominence1, prominence2)
            score = (
                (2.3 * norm_distance)
                + (0.20 * math.log(area_ratio))
                + (0.0025 * lab_distance)
                - (0.0015 * prominence)
            )
            possible.append((score, i, j, norm_distance, lab_distance, prominence))

    selected = []
    selected_i = set()
    selected_j = set()
    for _score, i, j, norm_distance, lab_distance, prominence in sorted(possible, key=lambda item: item[0]):
        if i in selected_i or j in selected_j:
            continue
        patch1 = patches1[i]
        patch2 = patches2[j]
        selected_i.add(i)
        selected_j.add(j)
        selected.append(
            LandmarkCandidate(
                match=None,
                point1=patch1.center,
                point2=patch2.center,
                descriptor_distance=lab_distance,
                reprojection_error=float(norm_distance * 100.0),
                stable_spot_score=min(patch1.score, patch2.score),
                prominence_score=float(prominence),
                red_brown_signal=min(patch1.score, patch2.score),
                box_half1=patch1.box_half,
                box_half2=patch2.box_half,
                source="prominent_patch",
            )
        )
        if len(selected) >= max_matches:
            break

    return selected


def detect_stable_spot_keypoints(
    image: np.ndarray,
    mask: np.ndarray,
    max_spots: int,
    min_score: float,
) -> List[cv2.KeyPoint]:
    """Find small dark/red-brown spot-like centers to force SIFT descriptors there."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgr = image.astype(np.float32)
    b, g, r = cv2.split(bgr)

    local_bg = cv2.GaussianBlur(gray, (0, 0), 9)
    dark_contrast = np.maximum(0.0, local_bg - gray)
    red_brown = np.maximum(0.0, r - (0.55 * g + 0.45 * b))
    score_map = (1.35 * dark_contrast) + (0.55 * red_brown)
    score_map[mask == 0] = 0

    roi_scores = score_map[mask > 0]
    if roi_scores.size == 0:
        return []

    threshold = max(float(np.percentile(roi_scores, 99.2)), min_score)
    candidates = ((score_map >= threshold) & (mask > 0)).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    spots = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 450 or max(w, h) > 36:
            continue

        x, y = centroids[label]
        measurement = stable_spot_measurement(image, (float(x), float(y)))
        point_score = measurement["score"]
        if point_score < min_score or measurement["skin_penalty"] > 0:
            continue
        spots.append((measurement["prominence"], area, float(x), float(y)))

    spots.sort(reverse=True)
    keypoints = []
    for score, area, x, y in spots[:max_spots]:
        size = float(np.clip(2.5 * math.sqrt(area), 8.0, 24.0))
        keypoints.append(cv2.KeyPoint(x, y, size, response=float(score)))
    return keypoints


def augment_with_stable_spot_keypoints(
    image: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    keypoints,
    descriptors,
    max_spots: int,
    min_score: float,
    use_rootsift: bool = True,
):
    spot_keypoints = detect_stable_spot_keypoints(image, mask, max_spots, min_score)
    if not spot_keypoints:
        return keypoints, descriptors, 0

    existing_points = [kp.pt for kp in keypoints]
    unique_spots = []
    for spot in spot_keypoints:
        if _far_enough(spot.pt, existing_points, 4.0):
            unique_spots.append(spot)
            existing_points.append(spot.pt)

    if not unique_spots:
        return keypoints, descriptors, 0

    sift = cv2.SIFT_create()
    computed_spots, spot_descriptors = sift.compute(gray, unique_spots)
    if spot_descriptors is None or not computed_spots:
        return keypoints, descriptors, 0
    if use_rootsift:
        spot_descriptors = to_rootsift(spot_descriptors)

    combined_keypoints = list(keypoints) + list(computed_spots)
    if descriptors is None:
        combined_descriptors = spot_descriptors
    else:
        combined_descriptors = np.vstack([descriptors, spot_descriptors])
    return combined_keypoints, combined_descriptors, len(computed_spots)


def detect_mole_candidates(
    image: np.ndarray,
    mask: np.ndarray,
    max_moles: int,
    min_score: float,
) -> List[MoleCandidate]:
    """
    Detect compact mole/freckle-like marks for constellation matching.

    This is intentionally stricter than the brown patch detector: it prefers
    small, round/oval, locally darker marks and rejects broad diffuse patches.
    """
    if np.count_nonzero(mask) == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_mask = build_skin_color_mask(image, mask)
    bgr = image.astype(np.float32)
    b, g, r = cv2.split(bgr)

    local_bg = cv2.GaussianBlur(gray, (0, 0), 11)
    dark_contrast = np.maximum(0.0, local_bg - gray)
    red_brown = np.maximum(0.0, r - (0.54 * g + 0.46 * b))
    score_map = (1.55 * dark_contrast) + (0.55 * red_brown)
    score_map[mask == 0] = 0

    roi_scores = score_map[mask > 0]
    if roi_scores.size == 0:
        return []

    threshold = max(float(np.percentile(roi_scores, 98.7)), min_score)
    candidate_mask = ((score_map >= threshold) & (mask > 0)).astype(np.uint8) * 255
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    candidates: List[MoleCandidate] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 260:
            continue
        if max(w, h) > 32 or min(w, h) < 2:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        fill_ratio = area / max(1.0, float(w * h))
        if aspect > 2.4 or fill_ratio < 0.18:
            continue

        cx, cy = centroids[label]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        y0, y1 = max(0, cy_i - 16), min(image.shape[0], cy_i + 17)
        x0, x1 = max(0, cx_i - 16), min(image.shape[1], cx_i + 17)
        if np.mean(skin_mask[y0:y1, x0:x1] > 0) < 0.72:
            continue

        measurement = stable_spot_measurement(image, (float(cx), float(cy)), inner_radius=4, outer_radius=18)
        if measurement["skin_penalty"] > 0:
            continue
        if measurement["score"] < min_score:
            continue

        component = labels == label
        mean_lab = tuple(float(v) for v in np.mean(lab[component], axis=0))
        box_half = int(np.clip(max(w, h) * 0.85 + 8, 10, 30))
        candidates.append(
            MoleCandidate(
                center=(float(cx), float(cy)),
                bbox=(x, y, w, h),
                area=area,
                mean_lab=mean_lab,
                score=float(measurement["score"]),
                prominence=float(measurement["prominence"]),
                box_half=box_half,
            )
        )

    candidates.sort(key=lambda mole: (mole.prominence, mole.score, mole.area), reverse=True)
    return candidates[:max_moles]


def _triangle_signature(points: Sequence[Point]) -> Optional[Tuple[Tuple[float, float], float]]:
    if len(points) != 3:
        return None
    distances = [
        math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]),
        math.hypot(points[0][0] - points[2][0], points[0][1] - points[2][1]),
        math.hypot(points[1][0] - points[2][0], points[1][1] - points[2][1]),
    ]
    longest = max(distances)
    shortest = min(distances)
    if shortest < 18.0 or longest < 45.0:
        return None
    ordered = sorted(distances)
    return (ordered[0] / longest, ordered[1] / longest), longest


def _ordered_triangle_indices(candidates: Sequence[MoleCandidate], indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
    pts = np.array([candidates[i].center for i in indices], dtype=np.float32)
    centroid = np.mean(pts, axis=0)
    angles = [math.atan2(float(pt[1] - centroid[1]), float(pt[0] - centroid[0])) for pt in pts]
    ordered_local = sorted(range(3), key=lambda idx: angles[idx])
    return tuple(indices[idx] for idx in ordered_local)


def _build_mole_triangles(
    candidates: Sequence[MoleCandidate],
    max_triangles: int,
) -> List[Tuple[Tuple[int, int, int], Tuple[float, float], float, float]]:
    triangles = []
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ordered = _ordered_triangle_indices(candidates, (i, j, k))
                pts = [candidates[index].center for index in ordered]
                signature = _triangle_signature(pts)
                if signature is None:
                    continue
                ratios, scale = signature
                area = abs(cv2.contourArea(np.array(pts, dtype=np.float32)))
                if area < 240.0:
                    continue
                prominence = min(candidates[index].prominence for index in ordered)
                triangles.append((ordered, ratios, scale, prominence))

    triangles.sort(key=lambda item: item[3] * item[2], reverse=True)
    return triangles[:max_triangles]


def match_mole_constellations(
    moles1: Sequence[MoleCandidate],
    moles2: Sequence[MoleCandidate],
    max_triangles: int,
    shape_tolerance: float,
    max_projected_distance: float,
    max_lab_distance: float,
    max_area_ratio: float,
    max_matches: int,
) -> List[LandmarkCandidate]:
    if len(moles1) < 3 or len(moles2) < 3:
        return []

    triangles1 = _build_mole_triangles(moles1, max_triangles)
    triangles2 = _build_mole_triangles(moles2, max_triangles)
    if not triangles1 or not triangles2:
        return []

    best_matches: List[LandmarkCandidate] = []
    best_score = float("inf")

    for idxs1, ratios1, _scale1, _prom1 in triangles1:
        pts1 = np.array([moles1[index].center for index in idxs1], dtype=np.float32)
        for idxs2, ratios2, _scale2, _prom2 in triangles2:
            shape_error = math.hypot(ratios1[0] - ratios2[0], ratios1[1] - ratios2[1])
            if shape_error > shape_tolerance:
                continue

            pts2 = np.array([moles2[index].center for index in idxs2], dtype=np.float32)
            affine_2x3 = cv2.getAffineTransform(pts1, pts2)
            affine = np.vstack([affine_2x3, [0.0, 0.0, 1.0]]).astype(np.float32)

            possible = []
            for i, mole1 in enumerate(moles1):
                projected = cv2.perspectiveTransform(
                    np.float32([[mole1.center]]).reshape(-1, 1, 2),
                    affine,
                ).reshape(2)
                for j, mole2 in enumerate(moles2):
                    spatial_error = math.hypot(projected[0] - mole2.center[0], projected[1] - mole2.center[1])
                    if spatial_error > max_projected_distance:
                        continue

                    lab_distance = float(np.linalg.norm(np.array(mole1.mean_lab) - np.array(mole2.mean_lab)))
                    if lab_distance > max_lab_distance:
                        continue

                    area_ratio = max(mole1.area, mole2.area) / max(1, min(mole1.area, mole2.area))
                    if area_ratio > max_area_ratio:
                        continue

                    pair_score = (
                        spatial_error
                        + (0.18 * lab_distance)
                        + (4.0 * math.log(area_ratio))
                        - (0.05 * min(mole1.prominence, mole2.prominence))
                    )
                    possible.append((pair_score, i, j, spatial_error, lab_distance))

            used1 = set()
            used2 = set()
            matches: List[LandmarkCandidate] = []
            for _pair_score, i, j, spatial_error, lab_distance in sorted(possible, key=lambda item: item[0]):
                if i in used1 or j in used2:
                    continue
                mole1 = moles1[i]
                mole2 = moles2[j]
                used1.add(i)
                used2.add(j)
                matches.append(
                    LandmarkCandidate(
                        match=None,
                        point1=mole1.center,
                        point2=mole2.center,
                        descriptor_distance=lab_distance,
                        reprojection_error=float(spatial_error),
                        stable_spot_score=min(mole1.score, mole2.score),
                        prominence_score=min(mole1.prominence, mole2.prominence),
                        red_brown_signal=min(mole1.score, mole2.score),
                        box_half1=mole1.box_half,
                        box_half2=mole2.box_half,
                        source="mole_constellation",
                    )
                )
                if len(matches) >= max_matches:
                    break

            if len(matches) < 3:
                continue
            avg_error = float(np.mean([match.reprojection_error for match in matches]))
            model_score = (-12.0 * len(matches)) + avg_error + (35.0 * shape_error)
            if len(matches) > len(best_matches) or (
                len(matches) == len(best_matches) and model_score < best_score
            ):
                best_matches = matches
                best_score = model_score

    return best_matches


def _mean_lab_patch(image: np.ndarray, center: Point, radius: int = 6) -> Tuple[float, float, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy = int(round(center[0])), int(round(center[1]))
    y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
    patch = lab[y0:y1, x0:x1]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in np.mean(patch.reshape(-1, 3), axis=0))


def _dedupe_stable_candidates(candidates: Sequence[StableCandidate], distance_threshold: float = 5.0) -> List[StableCandidate]:
    output: List[StableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.prominence * math.sqrt(max(1, item.area)), reverse=True):
        if any(math.hypot(candidate.center[0] - existing.center[0], candidate.center[1] - existing.center[1]) < distance_threshold for existing in output):
            continue
        output.append(candidate)
    return output


def detect_stable_candidates_for_geometry(
    image: np.ndarray,
    mask: np.ndarray,
    max_moles: int,
    min_mole_score: float,
    max_brown_patches: int,
    brown_patch_min_area: int,
    brown_patch_max_area: int,
    brown_patch_min_score: float,
    stable_spot_keypoints: int,
    min_stable_spot_score: float,
) -> List[StableCandidate]:
    candidates: List[StableCandidate] = []

    for mole in detect_mole_candidates(image, mask, max_moles, min_mole_score):
        candidates.append(
            StableCandidate(
                center=mole.center,
                area=mole.area,
                mean_lab=mole.mean_lab,
                score=mole.score,
                prominence=mole.prominence,
                box_half=mole.box_half,
                source="mole",
            )
        )

    for patch in detect_brown_patch_candidates(
        image,
        mask,
        max_brown_patches,
        brown_patch_min_area,
        brown_patch_max_area,
        brown_patch_min_score,
    ):
        candidates.append(
            StableCandidate(
                center=patch.center,
                area=patch.area,
                mean_lab=patch.mean_lab,
                score=patch.score,
                prominence=patch.score * math.sqrt(max(1, patch.area)),
                box_half=patch.box_half,
                source="brown_patch",
            )
        )

    for spot in detect_stable_spot_keypoints(image, mask, stable_spot_keypoints, min_stable_spot_score):
        measurement = stable_spot_measurement(image, spot.pt)
        candidates.append(
            StableCandidate(
                center=(float(spot.pt[0]), float(spot.pt[1])),
                area=max(3, int(round((spot.size / 2.0) ** 2))),
                mean_lab=_mean_lab_patch(image, spot.pt),
                score=float(spot.response),
                prominence=float(measurement["prominence"]),
                box_half=max(8, int(round(spot.size / 2.0))),
                source="stable_spot",
            )
        )

    return _dedupe_stable_candidates(candidates)


def _stable_candidate_pair_score(
    candidate1: StableCandidate,
    candidate2: StableCandidate,
    image1_shape: Tuple[int, int, int],
    image2_shape: Tuple[int, int, int],
) -> float:
    lab_distance = float(np.linalg.norm(np.array(candidate1.mean_lab) - np.array(candidate2.mean_lab)))
    area_ratio = max(candidate1.area, candidate2.area) / max(1, min(candidate1.area, candidate2.area))
    h1, w1 = image1_shape[:2]
    h2, w2 = image2_shape[:2]
    normalized_distance = math.hypot(
        (candidate1.center[0] / max(1, w1)) - (candidate2.center[0] / max(1, w2)),
        (candidate1.center[1] / max(1, h1)) - (candidate2.center[1] / max(1, h2)),
    )
    source_penalty = 0.0 if candidate1.source == candidate2.source else 0.18
    prominence_bonus = min(candidate1.prominence, candidate2.prominence) / 900.0
    return (
        (lab_distance / 70.0)
        + (0.48 * abs(math.log(area_ratio)))
        + (1.20 * normalized_distance)
        + source_penalty
        - (0.12 * prominence_bonus)
    )


def match_geometry_candidate_ransac(
    candidates1: Sequence[StableCandidate],
    candidates2: Sequence[StableCandidate],
    image1_shape: Tuple[int, int, int],
    image2_shape: Tuple[int, int, int],
    max_pairings_per_candidate: int,
    max_tentative_pairs: int,
    ransac_threshold: float,
    max_projected_distance: float,
    max_lab_distance: float,
    max_area_ratio: float,
    max_matches: int,
) -> Tuple[List[LandmarkCandidate], int]:
    if len(candidates1) < 3 or len(candidates2) < 3:
        return [], 0

    tentative = []
    for i, candidate1 in enumerate(candidates1):
        scored = []
        for j, candidate2 in enumerate(candidates2):
            area_ratio = max(candidate1.area, candidate2.area) / max(1, min(candidate1.area, candidate2.area))
            if area_ratio > max_area_ratio:
                continue
            lab_distance = float(np.linalg.norm(np.array(candidate1.mean_lab) - np.array(candidate2.mean_lab)))
            if lab_distance > max_lab_distance * 1.65:
                continue
            score = _stable_candidate_pair_score(candidate1, candidate2, image1_shape, image2_shape)
            scored.append((score, i, j, lab_distance, area_ratio))
        tentative.extend(sorted(scored, key=lambda item: item[0])[:max_pairings_per_candidate])

    tentative = sorted(tentative, key=lambda item: item[0])[:max_tentative_pairs]
    if len(tentative) < 3:
        return [], len(tentative)

    src = np.array([candidates1[i].center for _score, i, _j, _lab, _area in tentative], dtype=np.float32)
    dst = np.array([candidates2[j].center for _score, _i, j, _lab, _area in tentative], dtype=np.float32)
    affine_2x3, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=6000,
        confidence=0.995,
    )
    if affine_2x3 is None or inlier_mask is None:
        return [], len(tentative)

    affine = np.vstack([affine_2x3, [0.0, 0.0, 1.0]]).astype(np.float32)
    inlier_count = int(np.count_nonzero(inlier_mask))
    possible = []
    for i, candidate1 in enumerate(candidates1):
        projected = cv2.perspectiveTransform(np.float32([[candidate1.center]]).reshape(-1, 1, 2), affine).reshape(2)
        for j, candidate2 in enumerate(candidates2):
            spatial_error = math.hypot(projected[0] - candidate2.center[0], projected[1] - candidate2.center[1])
            if spatial_error > max_projected_distance:
                continue
            lab_distance = float(np.linalg.norm(np.array(candidate1.mean_lab) - np.array(candidate2.mean_lab)))
            if lab_distance > max_lab_distance:
                continue
            area_ratio = max(candidate1.area, candidate2.area) / max(1, min(candidate1.area, candidate2.area))
            if area_ratio > max_area_ratio:
                continue
            pair_score = (
                spatial_error
                + (0.16 * lab_distance)
                + (4.0 * abs(math.log(area_ratio)))
                - (0.015 * min(candidate1.prominence, candidate2.prominence))
            )
            possible.append((pair_score, i, j, spatial_error, lab_distance))

    selected: List[LandmarkCandidate] = []
    used1 = set()
    used2 = set()
    for _score, i, j, spatial_error, lab_distance in sorted(possible, key=lambda item: item[0]):
        if i in used1 or j in used2:
            continue
        candidate1 = candidates1[i]
        candidate2 = candidates2[j]
        used1.add(i)
        used2.add(j)
        selected.append(
            LandmarkCandidate(
                match=None,
                point1=candidate1.center,
                point2=candidate2.center,
                descriptor_distance=lab_distance,
                reprojection_error=float(spatial_error),
                stable_spot_score=min(candidate1.score, candidate2.score),
                prominence_score=min(candidate1.prominence, candidate2.prominence),
                red_brown_signal=min(candidate1.score, candidate2.score),
                box_half1=candidate1.box_half,
                box_half2=candidate2.box_half,
                source="geometry_candidate",
            )
        )
        if len(selected) >= max_matches:
            break

    return selected, inlier_count


def match_geometry_candidates_with_model(
    candidates1: Sequence[StableCandidate],
    candidates2: Sequence[StableCandidate],
    geometry: Optional[np.ndarray],
    max_projected_distance: float,
    max_lab_distance: float,
    max_area_ratio: float,
    max_matches: int,
) -> List[LandmarkCandidate]:
    if geometry is None or not candidates1 or not candidates2:
        return []

    try:
        inverse_geometry = np.linalg.inv(geometry)
    except np.linalg.LinAlgError:
        inverse_geometry = None

    possible = []
    for i, candidate1 in enumerate(candidates1):
        projected = cv2.perspectiveTransform(
            np.float32([[candidate1.center]]).reshape(-1, 1, 2),
            geometry,
        ).reshape(2)
        for j, candidate2 in enumerate(candidates2):
            spatial_error = math.hypot(projected[0] - candidate2.center[0], projected[1] - candidate2.center[1])
            if spatial_error > max_projected_distance:
                continue
            reverse_error = 0.0
            if inverse_geometry is not None:
                reverse_projected = cv2.perspectiveTransform(
                    np.float32([[candidate2.center]]).reshape(-1, 1, 2),
                    inverse_geometry,
                ).reshape(2)
                reverse_error = math.hypot(
                    reverse_projected[0] - candidate1.center[0],
                    reverse_projected[1] - candidate1.center[1],
                )
                if reverse_error > max_projected_distance:
                    continue
            lab_distance = float(np.linalg.norm(np.array(candidate1.mean_lab) - np.array(candidate2.mean_lab)))
            if lab_distance > max_lab_distance:
                continue
            area_ratio = max(candidate1.area, candidate2.area) / max(1, min(candidate1.area, candidate2.area))
            if area_ratio > max_area_ratio:
                continue

            same_source_bonus = 6.0 if candidate1.source == candidate2.source else 0.0
            prominence_bonus = 0.018 * min(candidate1.prominence, candidate2.prominence)
            compact_bonus = 2.0 if max(candidate1.box_half, candidate2.box_half) <= 28 else 0.0
            pair_score = (
                spatial_error
                + (0.65 * reverse_error)
                + (0.14 * lab_distance)
                + (4.5 * abs(math.log(area_ratio)))
                - same_source_bonus
                - prominence_bonus
                - compact_bonus
            )
            possible.append((pair_score, i, j, spatial_error, lab_distance))

    selected: List[LandmarkCandidate] = []
    used1 = set()
    used2 = set()
    for _score, i, j, spatial_error, lab_distance in sorted(possible, key=lambda item: item[0]):
        if i in used1 or j in used2:
            continue
        candidate1 = candidates1[i]
        candidate2 = candidates2[j]
        used1.add(i)
        used2.add(j)
        selected.append(
            LandmarkCandidate(
                match=None,
                point1=candidate1.center,
                point2=candidate2.center,
                descriptor_distance=lab_distance,
                reprojection_error=float(spatial_error),
                stable_spot_score=min(candidate1.score, candidate2.score),
                prominence_score=min(candidate1.prominence, candidate2.prominence),
                red_brown_signal=min(candidate1.score, candidate2.score),
                box_half1=candidate1.box_half,
                box_half2=candidate2.box_half,
                source="geometry_candidate",
            )
        )
        if len(selected) >= max_matches:
            break

    return selected


def select_spread_out_landmarks(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1,
    kp2,
    inlier_matches: Sequence[cv2.DMatch],
    reprojection_errors: Sequence[float],
    top: int,
    min_spacing: float,
    stable_spot_weight: float,
    min_selected_spot_score: float,
    min_prominence_score: float,
    min_red_brown_signal: float,
) -> List[LandmarkCandidate]:
    candidates = []
    if len(inlier_matches) == 0:
        return candidates

    distances = np.array([m.distance for m in inlier_matches], dtype=np.float32)
    errors = np.array(reprojection_errors, dtype=np.float32)
    spot_scores = []
    prominence_scores = []
    red_brown_signals = []
    measurements = []
    for match in inlier_matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        measurement1 = stable_spot_measurement(img1, p1)
        measurement2 = stable_spot_measurement(img2, p2)
        spot_scores.append(min(measurement1["score"], measurement2["score"]))
        prominence_scores.append(min(measurement1["prominence"], measurement2["prominence"]))
        red_brown_signals.append(min(measurement1["red_brown_signal"], measurement2["red_brown_signal"]))
        measurements.append((measurement1, measurement2))
    spots = np.array(spot_scores, dtype=np.float32)
    prominences = np.array(prominence_scores, dtype=np.float32)

    max_distance = float(np.max(distances)) if len(distances) else 1.0
    max_error = float(np.max(errors)) if len(errors) else 1.0
    max_spot = float(np.max(spots)) if len(spots) else 1.0
    max_prominence = float(np.max(prominences)) if len(prominences) else 1.0
    max_distance = max(max_distance, 1e-6)
    max_error = max(max_error, 1e-6)
    max_spot = max(max_spot, 1e-6)
    max_prominence = max(max_prominence, 1e-6)

    scored = []
    for match, error, spot_score, prominence_score, red_brown_signal, pair_measurements in zip(
        inlier_matches,
        reprojection_errors,
        spot_scores,
        prominence_scores,
        red_brown_signals,
        measurements,
    ):
        if spot_score < min_selected_spot_score or prominence_score < min_prominence_score:
            continue
        if red_brown_signal < min_red_brown_signal:
            continue

        measurement1, measurement2 = pair_measurements
        if measurement1["skin_penalty"] > 0 or measurement2["skin_penalty"] > 0:
            continue

        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        distance_score = float(match.distance) / max_distance
        error_score = float(error) / max_error
        stable_spot_bonus = float(spot_score) / max_spot
        prominence_bonus = float(prominence_score) / max_prominence
        score = (
            (0.42 * distance_score)
            + (0.26 * error_score)
            - (stable_spot_weight * stable_spot_bonus)
            - (0.35 * prominence_bonus)
        )
        scored.append(
            (
                score,
                LandmarkCandidate(
                    match,
                    p1,
                    p2,
                    float(match.distance),
                    float(error),
                    float(spot_score),
                    float(prominence_score),
                    float(red_brown_signal),
                    int(round(measurement1["box_half"])),
                    int(round(measurement2["box_half"])),
                ),
            )
        )

    selected: List[LandmarkCandidate] = []
    selected_pts1: List[Point] = []
    selected_pts2: List[Point] = []

    for _score, candidate in sorted(scored, key=lambda item: item[0]):
        if not _far_enough(candidate.point1, selected_pts1, min_spacing):
            continue
        if not _far_enough(candidate.point2, selected_pts2, min_spacing):
            continue
        selected.append(candidate)
        selected_pts1.append(candidate.point1)
        selected_pts2.append(candidate.point2)
        if len(selected) >= top:
            break

    return selected


def merge_landmark_candidates(
    primary: Sequence[LandmarkCandidate],
    extra: Sequence[LandmarkCandidate],
    top: int,
    min_spacing: float,
    patch_priority: float,
    max_selected_patch_error: float,
) -> List[LandmarkCandidate]:
    combined = list(primary) + [
        candidate
        for candidate in extra
        if candidate.source not in {"brown_patch", "prominent_patch"}
        or candidate.reprojection_error <= max_selected_patch_error
    ]

    def candidate_rank(candidate: LandmarkCandidate):
        if candidate.source == "brown_patch":
            source_bonus = patch_priority
        elif candidate.source == "mole_constellation":
            source_bonus = patch_priority * 1.35
        elif candidate.source == "geometry_candidate":
            source_bonus = patch_priority * 1.10
        elif candidate.source == "prominent_patch":
            source_bonus = patch_priority * 0.35
        else:
            source_bonus = 0.0
        return (
            -candidate.prominence_score - source_bonus,
            candidate.reprojection_error,
            candidate.descriptor_distance,
        )

    selected: List[LandmarkCandidate] = []
    selected_pts1: List[Point] = []
    selected_pts2: List[Point] = []
    for candidate in sorted(combined, key=candidate_rank):
        if not _far_enough(candidate.point1, selected_pts1, min_spacing):
            continue
        if not _far_enough(candidate.point2, selected_pts2, min_spacing):
            continue
        selected.append(candidate)
        selected_pts1.append(candidate.point1)
        selected_pts2.append(candidate.point2)
        if len(selected) >= top:
            break

    return selected


def filter_candidates_by_geometry(
    candidates: Sequence[LandmarkCandidate],
    geometry: Optional[np.ndarray],
    max_error: float,
) -> List[LandmarkCandidate]:
    if geometry is None:
        return list(candidates)
    output = []
    for candidate in candidates:
        projected = cv2.perspectiveTransform(
            np.float32([[candidate.point1]]).reshape(-1, 1, 2),
            geometry,
        ).reshape(2)
        error = math.hypot(projected[0] - candidate.point2[0], projected[1] - candidate.point2[1])
        if error <= max_error:
            candidate.reprojection_error = min(candidate.reprojection_error, float(error))
            output.append(candidate)
    return output


def _pad_to_same_height(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h1, _ = img1.shape[:2]
    h2, _ = img2.shape[:2]
    target_h = max(h1, h2)

    def pad(img: np.ndarray) -> np.ndarray:
        h, _ = img.shape[:2]
        if h == target_h:
            return img
        return cv2.copyMakeBorder(img, 0, target_h - h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return pad(img1), pad(img2)


def _label_text(image: np.ndarray, text: str, origin: Tuple[int, int], scale: float, color: Tuple[int, int, int]):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_labeled_landmarks(
    img1: np.ndarray,
    img2: np.ndarray,
    landmarks: Sequence[LandmarkCandidate],
    output_scale: float = 1.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if output_scale <= 0:
        raise ValueError("--output-scale must be > 0")

    if output_scale != 1.0:
        draw1 = cv2.resize(img1, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_AREA)
        draw2 = cv2.resize(img2, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_AREA)
    else:
        draw1 = img1.copy()
        draw2 = img2.copy()

    draw1, draw2 = _pad_to_same_height(draw1, draw2)
    h, w1 = draw1.shape[:2]
    combined = np.hstack([draw1, draw2])

    palette = [
        (0, 0, 255),
        (0, 128, 255),
        (0, 210, 255),
        (0, 190, 0),
        (255, 0, 0),
        (255, 0, 255),
        (170, 0, 255),
        (0, 255, 255),
        (90, 180, 60),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 128),
    ]

    rows: List[Dict[str, Any]] = []
    for index, landmark in enumerate(landmarks, start=1):
        color = palette[(index - 1) % len(palette)]
        x1, y1 = landmark.point1
        x2, y2 = landmark.point2
        x1s = int(round(x1 * output_scale))
        y1s = int(round(y1 * output_scale))
        x2s = int(round(x2 * output_scale)) + w1
        y2s = int(round(y2 * output_scale))

        box_half1 = max(8, int(round(landmark.box_half1 * output_scale)))
        box_half2 = max(8, int(round(landmark.box_half2 * output_scale)))
        for cx, cy, box_half in [(x1s, y1s, box_half1), (x2s, y2s, box_half2)]:
            x_left = max(0, cx - box_half)
            y_top = max(0, cy - box_half)
            x_right = min(combined.shape[1] - 1, cx + box_half)
            y_bottom = min(combined.shape[0] - 1, cy + box_half)
            cv2.rectangle(combined, (x_left, y_top), (x_right, y_bottom), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.rectangle(combined, (x_left, y_top), (x_right, y_bottom), color, 2, cv2.LINE_AA)

        _label_text(combined, str(index), (x1s + box_half1 + 5, max(24, y1s - box_half1 - 5)), 0.85, color)
        _label_text(combined, str(index), (x2s + box_half2 + 5, max(24, y2s - box_half2 - 5)), 0.85, color)

        rows.append(
            {
                "landmark_id": index,
                "image1_x": round(float(x1), 2),
                "image1_y": round(float(y1), 2),
                "image2_x": round(float(x2), 2),
                "image2_y": round(float(y2), 2),
                "descriptor_distance": round(float(landmark.descriptor_distance), 4),
                "reprojection_error": round(float(landmark.reprojection_error), 4),
                "stable_spot_score": round(float(landmark.stable_spot_score), 4),
                "prominence_score": round(float(landmark.prominence_score), 4),
                "red_brown_signal": round(float(landmark.red_brown_signal), 4),
                "image1_box_half_size": int(landmark.box_half1),
                "image2_box_half_size": int(landmark.box_half2),
                "source": landmark.source,
                "ransac_inlier": True,
            }
        )

    _label_text(combined, "Image 1 / baseline", (30, 45), 1.1, (255, 255, 255))
    _label_text(combined, "Image 2 / follow-up", (w1 + 30, 45), 1.1, (255, 255, 255))
    cv2.line(combined, (w1, 0), (w1, h), (30, 30, 30), 2)
    return combined, rows


def save_csv(csv_path: Path, rows: Sequence[Dict[str, Any]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "landmark_id",
        "image1_x",
        "image1_y",
        "image2_x",
        "image2_y",
        "descriptor_distance",
        "reprojection_error",
        "stable_spot_score",
        "prominence_score",
        "red_brown_signal",
        "image1_box_half_size",
        "image2_box_half_size",
        "source",
        "ransac_inlier",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(json_path: Path, metrics: Dict[str, Any], landmarks: Sequence[Dict[str, Any]], status: str, message: str):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "message": message,
        "reliability_metrics": metrics,
        "landmarks": list(landmarks),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _draw_keypoints(image: np.ndarray, keypoints, mask: Optional[np.ndarray] = None) -> np.ndarray:
    drawn = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if mask is not None:
        overlay = drawn.copy()
        overlay[mask == 0] = (0, 0, 80)
        drawn = cv2.addWeighted(overlay, 0.25, drawn, 0.75, 0)
    return drawn


def _draw_mole_candidates(image: np.ndarray, candidates: Sequence[MoleCandidate], mask: Optional[np.ndarray] = None) -> np.ndarray:
    drawn = image.copy()
    if mask is not None:
        overlay = drawn.copy()
        overlay[mask == 0] = (0, 0, 80)
        drawn = cv2.addWeighted(overlay, 0.25, drawn, 0.75, 0)
    for index, candidate in enumerate(candidates, start=1):
        cx, cy = int(round(candidate.center[0])), int(round(candidate.center[1]))
        cv2.circle(drawn, (cx, cy), max(5, candidate.box_half), (0, 255, 255), 2, cv2.LINE_AA)
        _label_text(drawn, str(index), (cx + candidate.box_half + 3, max(16, cy - candidate.box_half - 3)), 0.45, (0, 255, 255))
    return drawn


def save_debug_outputs(
    output_dir: Path,
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    kp1,
    kp2,
    good_matches: Sequence[cv2.DMatch],
    inlier_matches: Sequence[cv2.DMatch],
    mole_candidates1: Optional[Sequence[MoleCandidate]] = None,
    mole_candidates2: Optional[Sequence[MoleCandidate]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "image1_allowed_mask.png"), mask1)
    cv2.imwrite(str(output_dir / "image2_allowed_mask.png"), mask2)
    cv2.imwrite(str(output_dir / "image1_keypoints.jpg"), _draw_keypoints(img1, kp1, mask1))
    cv2.imwrite(str(output_dir / "image2_keypoints.jpg"), _draw_keypoints(img2, kp2, mask2))
    if mole_candidates1 is not None:
        cv2.imwrite(str(output_dir / "image1_mole_candidates.jpg"), _draw_mole_candidates(img1, mole_candidates1, mask1))
    if mole_candidates2 is not None:
        cv2.imwrite(str(output_dir / "image2_mole_candidates.jpg"), _draw_mole_candidates(img2, mole_candidates2, mask2))

    before = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        list(good_matches),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_dir / "all_good_matches_before_ransac.jpg"), before)

    inliers = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        list(inlier_matches),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_dir / "ransac_inlier_matches.jpg"), inliers)


def _project_ring_outline(H: np.ndarray, center: Tuple[float, float], radius: float, samples: int = 96) -> np.ndarray:
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    circle = np.float32([(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(circle, H).reshape(-1, 2)
    return np.round(projected).astype(np.int32)


def save_projected_ring_position(
    output_dir: Path,
    img2: np.ndarray,
    H: np.ndarray,
    center: Tuple[float, float],
    radius: float,
):

    output = img2.copy()
    center_arr = np.float32([[center]]).reshape(-1, 1, 2)
    projected_center = cv2.perspectiveTransform(center_arr, H).reshape(2)
    cx, cy = int(round(projected_center[0])), int(round(projected_center[1]))

    outline = _project_ring_outline(H, center, radius)
    cv2.polylines(output, [outline], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(output, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 34, 3, cv2.LINE_AA)
    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
    _label_text(output, "Projected original ring position", (max(10, cx + 16), max(35, cy - 16)), 0.8, (0, 255, 255))
    cv2.imwrite(str(output_dir / "projected_ring_position.jpg"), output)


def _metrics_dict(metrics: MatchQuality) -> Dict[str, Any]:
    return {
        "total_keypoints_image1": metrics.total_keypoints_image1,
        "total_keypoints_image2": metrics.total_keypoints_image2,
        "raw_matches": metrics.raw_matches,
        "good_matches_after_lowe": metrics.good_matches_after_lowe,
        "ransac_inliers": metrics.ransac_inliers,
        "inlier_ratio": round(float(metrics.inlier_ratio), 6),
        "average_reprojection_error": (
            None
            if metrics.average_reprojection_error is None
            else round(float(metrics.average_reprojection_error), 6)
        ),
        "geometric_model": metrics.geometric_model,
        "sift_contrast_image1": metrics.sift_contrast_image1,
        "sift_contrast_image2": metrics.sift_contrast_image2,
        "lowe_ratio_used": metrics.lowe_ratio_used,
        "spatially_gated_matches": metrics.spatially_gated_matches,
        "stable_spot_keypoints_image1": metrics.stable_spot_keypoints_image1,
        "stable_spot_keypoints_image2": metrics.stable_spot_keypoints_image2,
        "brown_patch_candidates_image1": metrics.brown_patch_candidates_image1,
        "brown_patch_candidates_image2": metrics.brown_patch_candidates_image2,
        "brown_patch_matches": metrics.brown_patch_matches,
        "mole_candidates_image1": metrics.mole_candidates_image1,
        "mole_candidates_image2": metrics.mole_candidates_image2,
        "mole_constellation_matches": metrics.mole_constellation_matches,
        "geometry_candidate_matches": metrics.geometry_candidate_matches,
        "geometry_candidate_inliers": metrics.geometry_candidate_inliers,
        "anchor_pairs": metrics.anchor_pairs,
        "anchor_average_error": (
            None
            if metrics.anchor_average_error is None
            else round(float(metrics.anchor_average_error), 6)
        ),
        "anchor_model": metrics.anchor_model,
        "use_rootsift": metrics.use_rootsift,
        "cross_check_matches": metrics.cross_check_matches,
        "homography_method": metrics.homography_method,
    }


def _print_metrics(metrics: MatchQuality):
    print(f"Image 1 keypoints: {metrics.total_keypoints_image1}")
    print(f"Image 2 keypoints: {metrics.total_keypoints_image2}")
    print(f"Raw FLANN matches: {metrics.raw_matches}")
    print(f"Good matches after Lowe ratio test: {metrics.good_matches_after_lowe}")
    print(f"RANSAC inliers: {metrics.ransac_inliers}")
    print(f"Inlier ratio: {metrics.inlier_ratio:.3f}")
    print(f"Geometric model: {metrics.geometric_model or 'n/a'}")
    if metrics.sift_contrast_image1 is not None and metrics.sift_contrast_image2 is not None:
        print(
            "SIFT contrast thresholds used: "
            f"image1={metrics.sift_contrast_image1:g}, image2={metrics.sift_contrast_image2:g}"
        )
    if metrics.lowe_ratio_used is not None:
        print(f"Lowe ratio used after adaptive retry: {metrics.lowe_ratio_used:.2f}")
    print(f"Spatially gated matches: {metrics.spatially_gated_matches}")
    print(
        "Extra stable-spot keypoints: "
        f"image1={metrics.stable_spot_keypoints_image1}, image2={metrics.stable_spot_keypoints_image2}"
    )

    print(
        "Brown patch candidates/matches: "
        f"image1={metrics.brown_patch_candidates_image1}, "
        f"image2={metrics.brown_patch_candidates_image2}, "
        f"matches={metrics.brown_patch_matches}"
    )
    print(
        "Mole candidates/constellation matches: "
        f"image1={metrics.mole_candidates_image1}, "
        f"image2={metrics.mole_candidates_image2}, "
        f"matches={metrics.mole_constellation_matches}"
    )
    print(
        "Geometry-first candidate matches: "
        f"matches={metrics.geometry_candidate_matches}, "
        f"RANSAC inliers={metrics.geometry_candidate_inliers}"
    )
    if metrics.anchor_pairs:
        if metrics.anchor_average_error is None:
            print(f"Trusted anchor pairs: {metrics.anchor_pairs} ({metrics.anchor_model or 'n/a'}, error=n/a)")
        else:
            print(
                "Trusted anchor pairs: "
                f"{metrics.anchor_pairs} ({metrics.anchor_model or 'n/a'}, "
                f"avg error={metrics.anchor_average_error:.3f} px)"
            )
    if metrics.average_reprojection_error is None:
        print("Average reprojection error: n/a")
    else:
        print(f"Average reprojection error: {metrics.average_reprojection_error:.3f} px")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Conservatively label stable corresponding SIFT skin landmarks between two visits."
    )
    parser.add_argument("--image1", required=True, type=Path, help="Path to baseline/Visit 1 image")
    parser.add_argument("--image2", required=True, type=Path, help="Path to follow-up/Visit 2 image")
    parser.add_argument("--image1-mask", type=Path, help="Optional saved binary allowed mask for image 1")
    parser.add_argument("--image2-mask", type=Path, help="Optional saved binary allowed mask for image 2")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output artifacts")
    parser.add_argument(
        "--anchor-file",
        type=Path,
        help="Optional JSON/CSV of trusted corresponding anchor pairs: image1_x,image1_y,image2_x,image2_y",
    )
    parser.add_argument(
        "--select-anchor-pairs",
        nargs="?",
        const=3,
        default=0,
        type=int,
        help="Interactively click trusted anchor pairs before automatic landmark filling; default count is 3",
    )
    parser.add_argument(
        "--save-anchor-file",
        type=Path,
        help="Optional path to save clicked anchor pairs as reusable JSON",
    )
    parser.add_argument(
        "--anchor-auto-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In anchor mode, try to add extra automatic landmarks beyond the trusted anchors; default is anchors only",
    )
    parser.add_argument(
        "--anchor-autofill-max-error",
        default=8.0,
        type=float,
        help="Maximum anchor-geometry projection error for auto-added landmarks in anchor mode",
    )
    parser.add_argument(
        "--anchor-autofill-max-descriptor-distance",
        default=30.0,
        type=float,
        help="Maximum color/descriptor distance for auto-added landmarks in anchor mode",
    )
    parser.add_argument("--top", default=6, type=int, help="Maximum number of landmarks to label")
    parser.add_argument("--lowe-ratio", default=0.70, type=float, help="Lowe ratio test threshold")
    parser.add_argument(
        "--use-rootsift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RootSIFT descriptors for histogram-appropriate matching; default enabled",
    )
    parser.add_argument(
        "--cross-check-matches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require mutual nearest-neighbor agreement after Lowe ratio test; default enabled",
    )
    parser.add_argument("--ransac-threshold", default=4.0, type=float, help="RANSAC reprojection threshold in pixels")
    parser.add_argument("--min-inliers", default=6, type=int, help="Minimum RANSAC inliers required")
    parser.add_argument("--min-inlier-ratio", default=0.05, type=float, help="Minimum RANSAC inlier ratio required")
    parser.add_argument(
        "--max-avg-reprojection-error",
        default=2.5,
        type=float,
        help="Fail if accepted inliers have a larger average reprojection error",
    )
    parser.add_argument(
        "--max-anchor-average-error",
        default=12.0,
        type=float,
        help="Fail anchor mode if 4+ trusted anchors disagree by more than this average reprojection error",
    )
    parser.add_argument(
        "--max-avg-descriptor-distance",
        default=260.0,
        type=float,
        help="Fail if selected landmarks have weak average SIFT descriptor agreement",
    )
    parser.add_argument(
        "--stable-spot-weight",
        default=0.85,
        type=float,
        help="How strongly to prioritize small mole/freckle-like dark spots when ranking RANSAC inliers",
    )
    parser.add_argument(
        "--stable-spot-keypoints",
        default=120,
        type=int,
        help="Maximum extra mole/freckle-like keypoints to add per image before matching",
    )
    parser.add_argument(
        "--min-stable-spot-score",
        default=22.0,
        type=float,
        help="Minimum local dark/red-brown score for extra stable-spot keypoints",
    )
    parser.add_argument(
        "--min-selected-stable-spot-score",
        default=18.0,
        type=float,
        help="Minimum stable-spot score required for a final selected landmark",
    )
    parser.add_argument(
        "--min-prominence-score",
        default=28.0,
        type=float,
        help="Minimum visibility/prominence score required for a final selected landmark",
    )
    parser.add_argument(
        "--min-red-brown-signal",
        default=12.0,
        type=float,
        help="Minimum red/brown skin-mark signal required for a final selected landmark",
    )
    parser.add_argument(
        "--enable-brown-patch-landmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect and match medium brown patch/blob landmarks in addition to SIFT point landmarks",
    )
    parser.add_argument(
        "--brown-patch-keypoints",
        default=60,
        type=int,
        help="Maximum medium brown patch candidates to consider per image",
    )
    parser.add_argument(
        "--brown-patch-min-area",
        default=18,
        type=int,
        help="Minimum area in pixels for medium brown patch candidates",
    )
    parser.add_argument(
        "--brown-patch-max-area",
        default=9000,
        type=int,
        help="Maximum area in pixels for medium brown patch candidates",
    )

    parser.add_argument(
        "--brown-patch-min-score",
        default=10.0,
        type=float,
        help="Minimum brown-patch color/contrast score",
    )
    parser.add_argument(
        "--brown-patch-max-distance",
        default=45.0,
        type=float,
        help="Maximum projected pixel distance for matching brown patches",
    )
    parser.add_argument(
        "--brown-patch-max-lab-distance",
        default=42.0,
        type=float,
        help="Maximum LAB color distance for matching brown patches",
    )
    parser.add_argument(
        "--brown-patch-max-area-ratio",
        default=5.0,
        type=float,
        help="Maximum size ratio between matched brown patches",
    )
    parser.add_argument(
        "--brown-patch-priority",
        default=55.0,
        type=float,
        help="Ranking bonus for accepted medium brown patch landmarks",
    )
    parser.add_argument(
        "--max-selected-brown-patch-error",
        default=24.0,
        type=float,
        help="Maximum projected pixel error for a brown patch match to appear in final labels",
    )
    parser.add_argument(
        "--enable-position-fallback",
        action="store_true",
        help="Allow prominent patch matches by normalized position when geometry rejects them; experimental",
    )
    parser.add_argument(
        "--enable-mole-constellation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: detect compact mole/freckle candidates and match their relative constellation geometry",
    )
    parser.add_argument(
        "--mole-candidates",
        default=45,
        type=int,
        help="Maximum compact mole candidates to consider per image",
    )
    parser.add_argument(
        "--min-mole-score",
        default=20.0,
        type=float,
        help="Minimum local dark/red-brown score for mole constellation candidates",
    )
    parser.add_argument(
        "--mole-max-triangles",
        default=550,
        type=int,
        help="Maximum high-prominence mole triangles to compare between visits",
    )
    parser.add_argument(
        "--mole-shape-tolerance",
        default=0.075,
        type=float,
        help="Maximum triangle side-ratio difference for mole constellation matching",
    )
    parser.add_argument(
        "--mole-max-distance",
        default=38.0,
        type=float,
        help="Maximum projected pixel distance for mole constellation pair voting",
    )
    parser.add_argument(
        "--mole-max-lab-distance",
        default=55.0,
        type=float,
        help="Maximum LAB color distance for mole constellation pair voting",
    )
    parser.add_argument(
        "--mole-max-area-ratio",
        default=5.0,
        type=float,
        help="Maximum size ratio for mole constellation pair voting",
    )
    parser.add_argument(
        "--min-mole-constellation-matches",
        default=3,
        type=int,
        help="Minimum mole constellation matches needed to support a weak SIFT alignment",
    )
    parser.add_argument(
        "--allow-mole-constellation-rescue",
        action="store_true",
        help="Experimental: allow mole constellation matches to pass reliability checks when SIFT/RANSAC is weak",
    )
    parser.add_argument(
        "--enable-geometry-candidate-matcher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: use stable candidate pairings plus affine RANSAC as an additional matcher",
    )
    parser.add_argument(
        "--geometry-candidate-pairings",
        default=8,
        type=int,
        help="Tentative candidate pairings to keep per image1 candidate for geometry-first matching",
    )
    parser.add_argument(
        "--geometry-candidate-max-pairs",
        default=650,
        type=int,
        help="Maximum tentative stable candidate pairs used by geometry-first RANSAC",
    )
    parser.add_argument(
        "--geometry-candidate-ransac-threshold",
        default=42.0,
        type=float,
        help="RANSAC threshold for geometry-first stable candidate matching",
    )
    parser.add_argument(
        "--geometry-candidate-max-distance",
        default=48.0,
        type=float,
        help="Maximum projected distance for final geometry-first candidate matches",
    )
    parser.add_argument(
        "--geometry-candidate-max-lab-distance",
        default=70.0,
        type=float,
        help="Maximum LAB distance for final geometry-first candidate matches",
    )
    parser.add_argument(
        "--geometry-candidate-max-area-ratio",
        default=7.0,
        type=float,
        help="Maximum area ratio for final geometry-first candidate matches",
    )
    parser.add_argument(
        "--allow-geometry-candidate-rescue",
        action="store_true",
        help="Experimental: allow geometry-first candidate matches to pass reliability checks when SIFT/RANSAC is weak",
    )
    parser.add_argument(
        "--min-geometry-candidate-inliers",
        default=6,
        type=int,
        help="Minimum geometry-first RANSAC inliers needed to support weak SIFT/RANSAC",
    )
    parser.add_argument("--min-spacing", default=60.0, type=float, help="Minimum spacing between labeled landmarks")
    parser.add_argument("--select-polygon-roi", action="store_true", help="Interactively select polygon skin ROI")
    parser.add_argument(
        "--select-exclusion-polygons",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-auto-quality-mask",
        action="store_true",
        help="Use only the ROI mask; disables automatic glare/shadow/hair-like cleanup",
    )
    parser.add_argument(
        "--disable-spatial-prior",
        action="store_true",
        help="Do not use ROI shape to spatially gate descriptor matches",
    )
    parser.add_argument(
        "--spatial-gate",
        default=140.0,
        type=float,
        help="Maximum pixel distance from ROI-shape-predicted correspondence for a tentative match",
    )
    parser.add_argument("--baseline-ring-center", nargs=2, type=float, metavar=("X", "Y"), help="Visit 1 ring center")
    parser.add_argument("--baseline-ring-radius", type=float, metavar="R", help="Visit 1 ring radius")
    parser.add_argument(
        "--baseline-ring-inner-radius",
        type=float,
        metavar="R",
        help="Visit 1 inner ring radius; if omitted, it is estimated from the outer radius",
    )
    parser.add_argument(
        "--ring-exclusion-padding",
        default=10.0,
        type=float,
        help="Extra pixels added around the ring annulus exclusion",
    )
    parser.add_argument(
        "--auto-detect-baseline-ring",
        action="store_true",
        help="Automatically detect a circular baseline ring and exclude only its metal annulus",
    )
    parser.add_argument("--project-ring", action="store_true", help="Project baseline ring position onto follow-up image")
    parser.add_argument("--save-debug", action="store_true", help="Save masks, keypoints, and match debug images")
    parser.add_argument("--display-scale", default=0.75, type=float, help="Scale for interactive ROI windows")
    parser.add_argument("--output-scale", default=1.0, type=float, help="Scale for side-by-side labeled output")
    parser.add_argument("--max-features", default=12000, type=int, help="Maximum SIFT features")
    parser.add_argument("--contrast-threshold", default=0.006, type=float, help="Starting SIFT contrast threshold")
    parser.add_argument(
        "--target-keypoints",
        default=800,
        type=int,
        help="Adaptive SIFT lowers contrast until each image has roughly this many candidates, if possible",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    anchors: List[AnchorPair] = []
    if args.anchor_file is not None:
        anchors.extend(load_anchor_pairs(args.anchor_file))
        print(f"Loaded {len(anchors)} trusted anchor pair(s) from {args.anchor_file}")
    if args.select_anchor_pairs:
        clicked_anchors = select_anchor_pairs_interactive(
            img1,
            img2,
            int(args.select_anchor_pairs),
            args.display_scale,
        )
        anchors.extend(clicked_anchors)
        print(f"Selected {len(clicked_anchors)} trusted anchor pair(s) interactively.")
    if anchors and (args.save_anchor_file is not None or args.select_anchor_pairs):
        anchor_save_path = args.save_anchor_file or (output_dir / "anchor_pairs.json")
        save_anchor_pairs(anchor_save_path, anchors)
        print(f"Saved trusted anchor pairs: {anchor_save_path}")

    if args.image1_mask is not None:
        mask1 = load_mask(args.image1_mask, img1.shape)
        print(f"Image 1 / baseline: using saved mask {args.image1_mask}")
    else:
        mask1 = build_allowed_mask(
            img1,
            args.select_polygon_roi,
            "Image 1 / baseline",
            args.display_scale,
            not args.disable_auto_quality_mask,
        )

    if args.image2_mask is not None:
        mask2 = load_mask(args.image2_mask, img2.shape)
        print(f"Image 2 / follow-up: using saved mask {args.image2_mask}")
    else:
        mask2 = build_allowed_mask(
            img2,
            args.select_polygon_roi,
            "Image 2 / follow-up",
            args.display_scale,
            not args.disable_auto_quality_mask,
        )

    if args.select_exclusion_polygons:
        print("Note: --select-exclusion-polygons is deprecated and ignored; automatic quality masking is used instead.")

    detected_ring = None
    if args.auto_detect_baseline_ring and (
        args.baseline_ring_center is None or args.baseline_ring_radius is None
    ):
        detected_ring = detect_largest_ring_circle(img1)
        if detected_ring is None:
            print("Image 1 / baseline: no circular ring was auto-detected.")
        else:
            detected_center, detected_radius = detected_ring
            args.baseline_ring_center = [detected_center[0], detected_center[1]]
            args.baseline_ring_radius = detected_radius
            print(
                "Image 1 / baseline: auto-detected ring "
                f"center=({detected_center[0]:.1f}, {detected_center[1]:.1f}), "
                f"outer radius={detected_radius:.1f}."
            )

    if args.baseline_ring_center is not None and args.baseline_ring_radius is not None:
        mask1, applied_inner_radius = suppress_ring_annulus_from_mask(
            mask1,
            (float(args.baseline_ring_center[0]), float(args.baseline_ring_center[1])),
            float(args.baseline_ring_radius),
            args.baseline_ring_inner_radius,
            args.ring_exclusion_padding,
        )
        print(
            "Image 1 / baseline: excluded ring annulus from landmark detection mask "
            f"(outer radius={float(args.baseline_ring_radius):.1f}, "
            f"inner radius={applied_inner_radius:.1f})."
        )

    gray1 = preprocess_gray(img1)
    gray2 = preprocess_gray(img2)

    print("Running adaptive SIFT inside allowed masks...")
    kp1, des1, contrast1 = detect_sift_features_adaptive(
        gray1,
        mask1,
        args.max_features,
        args.contrast_threshold,
        args.target_keypoints,
        args.use_rootsift,
    )
    kp2, des2, contrast2 = detect_sift_features_adaptive(
        gray2,
        mask2,
        args.max_features,
        args.contrast_threshold,
        args.target_keypoints,
        args.use_rootsift,
    )
    kp1, des1, spot_count1 = augment_with_stable_spot_keypoints(
        img1,
        gray1,
        mask1,
        kp1,
        des1,
        args.stable_spot_keypoints,
        args.min_stable_spot_score,
        args.use_rootsift,
    )
    kp2, des2, spot_count2 = augment_with_stable_spot_keypoints(
        img2,
        gray2,
        mask2,
        kp2,
        des2,
        args.stable_spot_keypoints,
        args.min_stable_spot_score,
        args.use_rootsift,
    )

    spatial_prior = None
    if not args.disable_spatial_prior:
        spatial_prior = compute_mask_prior_affine(mask1, mask2)

    print("Matching descriptors with FLANN + adaptive Lowe ratio + spatial gating...")
    raw_matches, good_matches, lowe_ratio_used = match_descriptors_adaptive(
        des1,
        des2,
        kp1,
        kp2,
        args.lowe_ratio,
        spatial_prior,
        args.spatial_gate,
        max(args.min_inliers * 3, 30),
        args.cross_check_matches,
    )

    print("Computing RANSAC geometry...")
    H, inlier_matches, _inlier_flags, model_name = compute_geometric_model_ransac(
        kp1,
        kp2,
        good_matches,
        args.ransac_threshold,
    )
    inlier_errors = compute_reprojection_errors(H, kp1, kp2, inlier_matches) if H is not None else []

    metrics = MatchQuality(
        total_keypoints_image1=len(kp1),
        total_keypoints_image2=len(kp2),
        raw_matches=len(raw_matches),
        good_matches_after_lowe=len(good_matches),
        ransac_inliers=len(inlier_matches),
        inlier_ratio=(len(inlier_matches) / len(good_matches)) if good_matches else 0.0,
        average_reprojection_error=(float(np.mean(inlier_errors)) if inlier_errors else None),
        geometric_model=model_name,
        sift_contrast_image1=contrast1,
        sift_contrast_image2=contrast2,
        lowe_ratio_used=lowe_ratio_used,
        spatially_gated_matches=len(good_matches),
        stable_spot_keypoints_image1=spot_count1,
        stable_spot_keypoints_image2=spot_count2,
        use_rootsift=args.use_rootsift,
        cross_check_matches=args.cross_check_matches,
        homography_method="USAC_MAGSAC" if _HOMOGRAPHY_RANSAC_METHOD == getattr(cv2, "USAC_MAGSAC", None) else "RANSAC",
    )
    anchor_H, anchor_model, anchor_average_error = compute_anchor_geometry(anchors)
    if anchors and anchor_H is None:
        print("Trusted anchors were provided, but at least 3 anchor pairs are needed to estimate geometry.")
    H_for_candidates = anchor_H if anchor_H is not None else H
    if anchor_H is not None:
        print(
            "Using trusted anchor geometry for automatic landmark filling "
            f"({anchor_model}, avg anchor error={anchor_average_error:.3f} px)."
        )
    metrics.anchor_pairs = len(anchors)
    metrics.anchor_model = anchor_model
    metrics.anchor_average_error = anchor_average_error

    sift_selected = select_spread_out_landmarks(
        img1,
        img2,
        kp1,
        kp2,
        inlier_matches,
        inlier_errors,
        args.top,
        args.min_spacing,
        args.stable_spot_weight,
        args.min_selected_stable_spot_score,
        args.min_prominence_score,
        args.min_red_brown_signal,
    )

    patch_matches: List[LandmarkCandidate] = []
    patch_candidates1: List[BrownPatchCandidate] = []
    patch_candidates2: List[BrownPatchCandidate] = []
    mole_matches: List[LandmarkCandidate] = []
    mole_candidates1: List[MoleCandidate] = []
    mole_candidates2: List[MoleCandidate] = []
    geometry_candidate_matches: List[LandmarkCandidate] = []
    geometry_candidate_inliers = 0
    if args.enable_mole_constellation:
        mole_candidates1 = detect_mole_candidates(
            img1,
            mask1,
            args.mole_candidates,
            args.min_mole_score,
        )
        mole_candidates2 = detect_mole_candidates(
            img2,
            mask2,
            args.mole_candidates,
            args.min_mole_score,
        )
        mole_matches = match_mole_constellations(
            mole_candidates1,
            mole_candidates2,
            args.mole_max_triangles,
            args.mole_shape_tolerance,
            args.mole_max_distance,
            args.mole_max_lab_distance,
            args.mole_max_area_ratio,
            args.top,
        )

    if args.enable_geometry_candidate_matcher or anchor_H is not None:
        stable_candidates1 = detect_stable_candidates_for_geometry(
            img1,
            mask1,
            args.mole_candidates,
            args.min_mole_score,
            args.brown_patch_keypoints,
            args.brown_patch_min_area,
            args.brown_patch_max_area,
            args.brown_patch_min_score,
            args.stable_spot_keypoints,
            args.min_stable_spot_score,
        )
        stable_candidates2 = detect_stable_candidates_for_geometry(
            img2,
            mask2,
            args.mole_candidates,
            args.min_mole_score,
            args.brown_patch_keypoints,
            args.brown_patch_min_area,
            args.brown_patch_max_area,
            args.brown_patch_min_score,
            args.stable_spot_keypoints,
            args.min_stable_spot_score,
        )
        if H_for_candidates is not None:
            geometry_candidate_matches = match_geometry_candidates_with_model(
                stable_candidates1,
                stable_candidates2,
                H_for_candidates,
                args.geometry_candidate_max_distance,
                args.geometry_candidate_max_lab_distance,
                args.geometry_candidate_max_area_ratio,
                args.top,
            )
            geometry_candidate_inliers = len(inlier_matches)
        else:
            geometry_candidate_matches, geometry_candidate_inliers = match_geometry_candidate_ransac(
                stable_candidates1,
                stable_candidates2,
                img1.shape,
                img2.shape,
                args.geometry_candidate_pairings,
                args.geometry_candidate_max_pairs,
                args.geometry_candidate_ransac_threshold,
                args.geometry_candidate_max_distance,
                args.geometry_candidate_max_lab_distance,
                args.geometry_candidate_max_area_ratio,
                args.top,
            )

    if args.enable_brown_patch_landmarks and H_for_candidates is not None:
        patch_candidates1 = detect_brown_patch_candidates(
            img1,
            mask1,
            args.brown_patch_keypoints,
            args.brown_patch_min_area,
            args.brown_patch_max_area,
            args.brown_patch_min_score,
        )
        patch_candidates2 = detect_brown_patch_candidates(
            img2,
            mask2,
            args.brown_patch_keypoints,
            args.brown_patch_min_area,
            args.brown_patch_max_area,
            args.brown_patch_min_score,
        )
        patch_matches = match_brown_patch_candidates(
            patch_candidates1,
            patch_candidates2,
            H_for_candidates,
            args.brown_patch_max_distance,
            args.brown_patch_max_lab_distance,
            args.brown_patch_max_area_ratio,
        )
        if args.enable_position_fallback:
            patch_matches.extend(
                match_prominent_patches_by_position(
                    patch_candidates1,
                    patch_candidates2,
                    mask1,
                    mask2,
                    patch_matches,
                )
            )

    metrics.brown_patch_candidates_image1 = len(patch_candidates1)
    metrics.brown_patch_candidates_image2 = len(patch_candidates2)
    metrics.brown_patch_matches = len(patch_matches)
    metrics.mole_candidates_image1 = len(mole_candidates1)
    metrics.mole_candidates_image2 = len(mole_candidates2)
    metrics.mole_constellation_matches = len(mole_matches)
    metrics.geometry_candidate_matches = len(geometry_candidate_matches)
    metrics.geometry_candidate_inliers = geometry_candidate_inliers

    anchor_landmarks = [_anchor_to_candidate(anchor, index) for index, anchor in enumerate(anchors)]
    if anchor_H is not None:
        if not args.anchor_auto_fill:
            sift_selected = []
            geometry_candidate_matches = []
            mole_matches = []
            patch_matches = []
        else:
            strict_anchor_error = min(
                args.anchor_autofill_max_error,
                args.geometry_candidate_max_distance,
                args.brown_patch_max_distance,
            )
            geometry_candidate_matches = [
                candidate
                for candidate in geometry_candidate_matches
                if (
                    candidate.reprojection_error <= strict_anchor_error
                    and candidate.descriptor_distance <= args.anchor_autofill_max_descriptor_distance
                )
            ]
            patch_matches = [
                candidate
                for candidate in patch_matches
                if (
                    candidate.reprojection_error <= strict_anchor_error
                    and candidate.descriptor_distance <= args.anchor_autofill_max_descriptor_distance
                )
            ]
            mole_matches = [
                candidate
                for candidate in mole_matches
                if (
                    candidate.reprojection_error <= strict_anchor_error
                    and candidate.descriptor_distance <= args.anchor_autofill_max_descriptor_distance
                )
            ]
        sift_selected = filter_candidates_by_geometry(
            sift_selected,
            anchor_H,
            args.anchor_autofill_max_error if args.anchor_auto_fill else 0.0,
        )
        metrics.brown_patch_matches = len(patch_matches)
        metrics.mole_constellation_matches = len(mole_matches)
        metrics.geometry_candidate_matches = len(geometry_candidate_matches)

    selected = merge_landmark_candidates(
        list(anchor_landmarks) + sift_selected,
        list(geometry_candidate_matches) + list(mole_matches) + patch_matches,
        args.top,
        args.min_spacing,
        args.brown_patch_priority,
        args.max_selected_brown_patch_error,
    )
    metrics_payload = _metrics_dict(metrics)

    if args.save_debug:
        save_debug_outputs(
            output_dir,
            img1,
            img2,
            mask1,
            mask2,
            kp1,
            kp2,
            good_matches,
            inlier_matches,
            mole_candidates1,
            mole_candidates2,
        )

    _print_metrics(metrics)
    print(f"Selected landmarks for labeling: {len(selected)}")

    fail_reasons = []
    has_mole_constellation_support = (
        args.allow_mole_constellation_rescue
        and metrics.mole_constellation_matches >= args.min_mole_constellation_matches
    )
    has_geometry_candidate_support = (
        args.allow_geometry_candidate_rescue
        and metrics.geometry_candidate_matches >= args.min_inliers
        and metrics.geometry_candidate_inliers >= args.min_geometry_candidate_inliers
    )
    has_anchor_support = (
        metrics.anchor_pairs >= 3
        and anchor_H is not None
        and (
            metrics.anchor_pairs <= 3
            or metrics.anchor_average_error is None
            or metrics.anchor_average_error <= args.max_anchor_average_error
        )
    )
    has_alternate_geometry_support = has_mole_constellation_support or has_geometry_candidate_support or has_anchor_support
    if 0 < metrics.anchor_pairs < 3:
        fail_reasons.append(f"trusted anchor pairs below minimum ({metrics.anchor_pairs} < 3)")
    if (
        metrics.anchor_pairs > 3
        and metrics.anchor_average_error is not None
        and metrics.anchor_average_error > args.max_anchor_average_error
    ):
        fail_reasons.append(
            "trusted anchors disagree above threshold "
            f"({metrics.anchor_average_error:.3f} > {args.max_anchor_average_error:.3f})"
        )
    if H is None and not has_alternate_geometry_support:
        fail_reasons.append("geometric model is None")
    if metrics.ransac_inliers < args.min_inliers and not has_alternate_geometry_support:
        fail_reasons.append(f"RANSAC inliers below threshold ({metrics.ransac_inliers} < {args.min_inliers})")
    if metrics.inlier_ratio < args.min_inlier_ratio and not has_alternate_geometry_support:
        fail_reasons.append(f"inlier ratio below threshold ({metrics.inlier_ratio:.3f} < {args.min_inlier_ratio:.3f})")
    if (
        metrics.average_reprojection_error is not None
        and metrics.average_reprojection_error > args.max_avg_reprojection_error
        and not has_alternate_geometry_support
    ):
        fail_reasons.append(
            "average reprojection error above threshold "
            f"({metrics.average_reprojection_error:.3f} > {args.max_avg_reprojection_error:.3f})"
        )
    if (
        metrics.ransac_inliers < args.min_inliers
        and metrics.mole_constellation_matches > 0
        and not has_mole_constellation_support
    ):
        fail_reasons.append(
            "mole constellation support below threshold "
            f"({metrics.mole_constellation_matches} < {args.min_mole_constellation_matches})"
        )
    if len(selected) == 0:
        fail_reasons.append("selected final landmarks = 0")
    if selected:
        avg_descriptor_distance = float(np.mean([landmark.descriptor_distance for landmark in selected]))
        if avg_descriptor_distance > args.max_avg_descriptor_distance:
            fail_reasons.append(
                "selected landmarks have weak descriptor agreement "
                f"({avg_descriptor_distance:.3f} > {args.max_avg_descriptor_distance:.3f})"
            )

    if fail_reasons:
        print("\n" + FAIL_MESSAGE)
        print("Fail reasons:")
        for reason in fail_reasons:
            print(f"- {reason}")

        save_csv(output_dir / "landmark_matches.csv", [])
        save_json(output_dir / "landmark_matches.json", metrics_payload, [], "failed", FAIL_MESSAGE)
        print(f"Saved reliability JSON: {output_dir / 'landmark_matches.json'}")
        print(f"Saved empty landmark CSV: {output_dir / 'landmark_matches.csv'}")
        return 0

    labeled, rows = draw_labeled_landmarks(img1, img2, selected, args.output_scale)
    cv2.imwrite(str(output_dir / "labeled_landmarks.jpg"), labeled)
    save_csv(output_dir / "landmark_matches.csv", rows)
    save_json(output_dir / "landmark_matches.json", metrics_payload, rows, "success", "Reliable landmarks selected.")

    if args.project_ring:
        if H_for_candidates is None:
            print("Skipping ring projection: no geometric model is available.")
        elif args.baseline_ring_center is None or args.baseline_ring_radius is None:
            print("Skipping ring projection: provide --baseline-ring-center x y and --baseline-ring-radius r.")
        else:
            save_projected_ring_position(
                output_dir,
                img2,
                H_for_candidates,
                (float(args.baseline_ring_center[0]), float(args.baseline_ring_center[1])),
                float(args.baseline_ring_radius),
            )
            print(f"Saved projected ring position: {output_dir / 'projected_ring_position.jpg'}")

    print(f"Saved labeled landmarks image: {output_dir / 'labeled_landmarks.jpg'}")
    print(f"Saved landmark CSV: {output_dir / 'landmark_matches.csv'}")
    print(f"Saved landmark JSON: {output_dir / 'landmark_matches.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

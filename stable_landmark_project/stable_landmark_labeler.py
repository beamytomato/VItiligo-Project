"""
stable_landmark_labeler.py

Purpose:
    Label corresponding stable landmark candidates between two skin images using
    SIFT feature matching + RANSAC geometric filtering.

Example:
    python stable_landmark_labeler.py \
        --image1 images/visit1.jpg \
        --image2 images/visit2.jpg \
        --select-roi \
        --top 15

Outputs:
    outputs/labeled_landmarks.jpg
    outputs/landmark_matches.csv

Notes:
    - SIFT finds candidate landmarks.
    - RANSAC removes inconsistent matches.
    - The program labels corresponding points with the same number in both images.
    - Use ROI selection to exclude background, glare, ring-only areas, and unstable vitiligo borders.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
ROI = Tuple[int, int, int, int]


# -----------------------------
# Utility functions
# -----------------------------
def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def preprocess_for_sift(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale and improve local contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def select_roi_scaled(window_name: str, image: np.ndarray, display_scale: float) -> Optional[ROI]:
    """
    Let the user draw an ROI on a scaled copy of the image.
    Returns ROI coordinates in the original image coordinate system.
    """
    if display_scale <= 0:
        raise ValueError("display_scale must be > 0")

    display = cv2.resize(image, None, fx=display_scale, fy=display_scale)
    print(f"Draw ROI for {window_name}. Press ENTER/SPACE to confirm, or C to cancel.")
    roi_scaled = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi_scaled
    if w == 0 or h == 0:
        return None

    x_orig = int(round(x / display_scale))
    y_orig = int(round(y / display_scale))
    w_orig = int(round(w / display_scale))
    h_orig = int(round(h / display_scale))

    return x_orig, y_orig, w_orig, h_orig


def make_roi_mask(image_shape: Tuple[int, int, int], roi: Optional[ROI]) -> Optional[np.ndarray]:
    """Create a binary mask for SIFT so it only detects features inside the ROI."""
    if roi is None:
        return None

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = roi
    x2 = max(0, min(w, x + rw))
    y2 = max(0, min(h, y + rh))
    x = max(0, min(w, x))
    y = max(0, min(h, y))
    mask[y:y2, x:x2] = 255
    return mask


def draw_roi(image: np.ndarray, roi: Optional[ROI], color=(0, 255, 255)) -> np.ndarray:
    output = image.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
    return output


def compute_sift(image: np.ndarray, mask: Optional[np.ndarray], max_features: int, contrast_threshold: float):
    gray = preprocess_for_sift(image)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors


def match_sift(des1: np.ndarray, des2: np.ndarray, lowe_ratio: float):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)
    return good


def ransac_filter(kp1, kp2, matches, reprojection_threshold: float):
    if len(matches) < 4:
        return None, [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojection_threshold)

    if H is None or mask is None:
        return None, [], None

    mask_flat = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, mask_flat) if keep]
    return H, inlier_matches, mask_flat


def far_enough(pt: Point, selected_pts: List[Point], min_spacing: float) -> bool:
    for prev in selected_pts:
        dx = pt[0] - prev[0]
        dy = pt[1] - prev[1]
        if math.hypot(dx, dy) < min_spacing:
            return False
    return True


def choose_landmarks(kp1, kp2, inlier_matches, top_n: int, min_spacing: float):
    """
    Choose a smaller set of easy-to-read landmarks.
    - Sorts by descriptor distance.
    - Avoids landmarks that cluster too closely together.
    """
    sorted_matches = sorted(inlier_matches, key=lambda m: m.distance)

    selected = []
    selected_pts1: List[Point] = []
    selected_pts2: List[Point] = []

    for m in sorted_matches:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt

        if not far_enough(p1, selected_pts1, min_spacing):
            continue
        if not far_enough(p2, selected_pts2, min_spacing):
            continue

        selected.append(m)
        selected_pts1.append(p1)
        selected_pts2.append(p2)

        if len(selected) >= top_n:
            break

    return selected


def pad_to_same_height(img1: np.ndarray, img2: np.ndarray):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h = max(h1, h2)

    def pad(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        pad_h = target_h - h
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return pad(img1, max_h), pad(img2, max_h)


def draw_labeled_matches(img1, img2, kp1, kp2, selected_matches, output_scale: float):
    """Create a side-by-side image with matching landmarks labeled by number."""
    if output_scale != 1.0:
        img1_draw = cv2.resize(img1, None, fx=output_scale, fy=output_scale)
        img2_draw = cv2.resize(img2, None, fx=output_scale, fy=output_scale)
    else:
        img1_draw = img1.copy()
        img2_draw = img2.copy()

    img1_draw, img2_draw = pad_to_same_height(img1_draw, img2_draw)
    h1, w1 = img1_draw.shape[:2]
    combined = np.hstack([img1_draw, img2_draw])

    # Repeatable palette for landmark labels.
    palette = [
        (0, 0, 255), (0, 128, 255), (0, 200, 255), (0, 255, 0),
        (255, 0, 0), (255, 0, 255), (180, 0, 255), (0, 255, 255),
        (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 128),
    ]

    rows = []

    for i, match in enumerate(selected_matches, start=1):
        color = palette[(i - 1) % len(palette)]

        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        x1s, y1s = int(round(x1 * output_scale)), int(round(y1 * output_scale))
        x2s, y2s = int(round(x2 * output_scale)) + w1, int(round(y2 * output_scale))

        # Draw points.
        cv2.circle(combined, (x1s, y1s), 10, color, -1)
        cv2.circle(combined, (x2s, y2s), 10, color, -1)

        # Draw label text with black outline for readability.
        for dx, dy, col, thick in [(14, -14, (0, 0, 0), 5), (14, -14, color, 2)]:
            cv2.putText(combined, str(i), (x1s + dx, y1s + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, thick)
            cv2.putText(combined, str(i), (x2s + dx, y2s + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, thick)

        # Draw line connecting corresponding landmarks.
        cv2.line(combined, (x1s, y1s), (x2s, y2s), color, 2)

        rows.append({
            "landmark_id": i,
            "image1_x": round(x1, 2),
            "image1_y": round(y1, 2),
            "image2_x": round(x2, 2),
            "image2_y": round(y2, 2),
            "descriptor_distance": round(float(match.distance), 4),
        })

    # Add headings.
    cv2.putText(combined, "Image 1 / baseline", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
    cv2.putText(combined, "Image 1 / baseline", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(combined, "Image 2 / follow-up", (w1 + 30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
    cv2.putText(combined, "Image 2 / follow-up", (w1 + 30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    return combined, rows


def save_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["landmark_id", "image1_x", "image1_y", "image2_x", "image2_y", "descriptor_distance"]
        )
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Label corresponding stable SIFT landmarks between two skin images.")
    parser.add_argument("--image1", required=True, type=Path, help="Path to baseline/visit 1 image")
    parser.add_argument("--image2", required=True, type=Path, help="Path to follow-up/visit 2 image")
    parser.add_argument("--output", default=Path("outputs/labeled_landmarks.jpg"), type=Path, help="Output labeled image path")
    parser.add_argument("--csv", default=Path("outputs/landmark_matches.csv"), type=Path, help="Output CSV path")
    parser.add_argument("--top", default=15, type=int, help="Number of labeled landmarks to show")
    parser.add_argument("--max-features", default=5000, type=int, help="Maximum SIFT features")
    parser.add_argument("--contrast-threshold", default=0.01, type=float, help="Lower values detect more SIFT features")
    parser.add_argument("--lowe-ratio", default=0.75, type=float, help="Lowe ratio test threshold")
    parser.add_argument("--ransac-threshold", default=5.0, type=float, help="RANSAC reprojection threshold in pixels")
    parser.add_argument("--min-inliers", default=10, type=int, help="Minimum RANSAC inliers required")
    parser.add_argument("--min-spacing", default=60, type=float, help="Minimum spacing between selected landmarks in pixels")
    parser.add_argument("--select-roi", action="store_true", help="Interactively select ROI in each image")
    parser.add_argument("--display-scale", default=0.5, type=float, help="Scale used for ROI selection windows")
    parser.add_argument("--output-scale", default=0.5, type=float, help="Scale for saved side-by-side output image")
    args = parser.parse_args()

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    roi1 = None
    roi2 = None
    if args.select_roi:
        roi1 = select_roi_scaled("Select ROI in Image 1", img1, args.display_scale)
        roi2 = select_roi_scaled("Select ROI in Image 2", img2, args.display_scale)
        print(f"ROI 1: {roi1}")
        print(f"ROI 2: {roi2}")

    mask1 = make_roi_mask(img1.shape, roi1)
    mask2 = make_roi_mask(img2.shape, roi2)

    print("Running SIFT...")
    kp1, des1 = compute_sift(img1, mask1, args.max_features, args.contrast_threshold)
    kp2, des2 = compute_sift(img2, mask2, args.max_features, args.contrast_threshold)

    print(f"Image 1 keypoints: {len(kp1)}")
    print(f"Image 2 keypoints: {len(kp2)}")

    if des1 is None or des2 is None:
        raise RuntimeError("SIFT did not find descriptors in one or both images. Try a larger ROI or sharper images.")

    print("Matching SIFT descriptors...")
    good_matches = match_sift(des1, des2, args.lowe_ratio)
    print(f"Good matches after Lowe ratio test: {len(good_matches)}")

    print("Running RANSAC...")
    H, inlier_matches, _ = ransac_filter(kp1, kp2, good_matches, args.ransac_threshold)
    print(f"RANSAC inliers: {len(inlier_matches)} / {len(good_matches)}")

    if H is None or len(inlier_matches) < args.min_inliers:
        print("\nWARNING: Not enough reliable RANSAC inliers.")
        print("This usually means the two images do not share enough stable visible landmarks,")
        print("or the ROI/background/lighting/viewpoint is too different.")
        print("Try selecting a smaller overlapping ROI, taking wider baseline photos, or improving lighting.\n")

    selected = choose_landmarks(kp1, kp2, inlier_matches, args.top, args.min_spacing)
    print(f"Selected landmarks for labeling: {len(selected)}")

    if len(selected) == 0:
        raise RuntimeError("No landmarks selected. Try lowering --min-spacing or improving image overlap.")

    labeled, rows = draw_labeled_matches(img1, img2, kp1, kp2, selected, args.output_scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), labeled)
    save_csv(args.csv, rows)

    print(f"Saved labeled landmarks image: {args.output}")
    print(f"Saved landmark CSV: {args.csv}")

    print("\nDone. Inspect the output image manually.")
    print("Only use landmarks that a human can confirm are real stable skin landmarks.")


if __name__ == "__main__":
    main()

from pathlib import Path
import cv2
import numpy as np
import math

# -----------------------------
# Paths
# -----------------------------
BASELINE_PATH = Path("images/baseline.jpg")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
CAMERA_INDEX = 0

# Change this to the original ring center in your baseline image.
# Format: (x, y)
BASELINE_RING_CENTER = (640, 420)

# Change this to the approximate ring radius in pixels from the baseline image.
BASELINE_RING_RADIUS = 180

# SIFT / matching settings
MAX_FEATURES = 3000
LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 20
RANSAC_REPROJECTION_THRESHOLD = 5.0

# Guide settings
ALIGNMENT_THRESHOLD_PIXELS = 25

# -----------------------------
# Helper function: detect current ring
# -----------------------------
def detect_ring(frame):
    """
    Detects the current metallic ring in the live frame.

    This is a simple starter version using grayscale thresholding,
    blur, edges, and contour fitting.

    Returns:
        center: (x, y) or None
        radius: float or None
        debug_frame: frame with ring detection drawn
    """

    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur helps reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological closing connects broken ring edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, output

    best_contour = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0

        # Ring should be somewhat circular/elliptical
        if 0.5 < circularity < 1.3 and 0.6 < aspect_ratio < 1.6:
            score = area * circularity

            if score > best_score:
                best_score = score
                best_contour = contour

    if best_contour is None:
        return None, None, output

    (x, y), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(output, center, radius, (255, 0, 0), 2)
    cv2.circle(output, center, 6, (255, 0, 0), -1)

    return center, radius, output


# -----------------------------
# Helper function: draw arrow
# -----------------------------
def draw_guidance(frame, current_center, target_center):
    """
    Draws movement guidance from current ring center to target ring center.
    """

    output = frame.copy()

    cx, cy = current_center
    tx, ty = target_center

    dx = tx - cx
    dy = ty - cy
    error = math.sqrt(dx * dx + dy * dy)

    # Draw target center
    cv2.circle(output, target_center, 10, (0, 255, 0), -1)
    cv2.putText(
        output,
        "TARGET",
        (tx + 12, ty - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Draw current center
    cv2.circle(output, current_center, 10, (255, 0, 0), -1)
    cv2.putText(
        output,
        "CURRENT",
        (cx + 12, cy + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # Draw arrow from current to target
    cv2.arrowedLine(
        output,
        current_center,
        target_center,
        (0, 255, 255),
        4,
        tipLength=0.25
    )

    # Text instructions
    instruction = f"Move ring: dx={dx:+.0f}px, dy={dy:+.0f}px | error={error:.1f}px"

    if error <= ALIGNMENT_THRESHOLD_PIXELS:
        status = "ALIGNED - hold position"
        status_color = (0, 255, 0)
    else:
        status = "NOT ALIGNED"
        status_color = (0, 0, 255)

    cv2.putText(
        output,
        instruction,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        status,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        3
    )

    return output, error


# -----------------------------
# Load baseline image
# -----------------------------
baseline = cv2.imread(str(BASELINE_PATH))

if baseline is None:
    raise FileNotFoundError(f"Could not load baseline image: {BASELINE_PATH}")

baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
baseline_gray = cv2.equalizeHist(baseline_gray)

# -----------------------------
# Create SIFT detector
# -----------------------------
sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)

# Detect baseline features once
baseline_keypoints, baseline_descriptors = sift.detectAndCompute(baseline_gray, None)

if baseline_descriptors is None:
    raise ValueError("No SIFT descriptors found in baseline image.")

print(f"Baseline keypoints: {len(baseline_keypoints)}")

# -----------------------------
# Create FLANN matcher for SIFT descriptors
# -----------------------------
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# -----------------------------
# Start camera
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try CAMERA_INDEX = 1 or 2.")

print("Live guide started.")
print("Press Q to quit.")
print("Press S to save the current frame.")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    display = frame.copy()

    # -----------------------------
    # Detect current ring in live frame
    # -----------------------------
    current_ring_center, current_ring_radius, ring_debug_frame = detect_ring(frame)

    # -----------------------------
    # SIFT on live frame
    # -----------------------------
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    frame_keypoints, frame_descriptors = sift.detectAndCompute(frame_gray, None)

    target_ring_center = None
    match_status = "Matching..."

    if frame_descriptors is not None and len(frame_keypoints) >= 10:
        matches = flann.knnMatch(baseline_descriptors, frame_descriptors, k=2)

        good_matches = []

        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= MIN_GOOD_MATCHES:
            src_pts = np.float32([
                baseline_keypoints[m.queryIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)

            dst_pts = np.float32([
                frame_keypoints[m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                RANSAC_REPROJECTION_THRESHOLD
            )

            if H is not None and mask is not None:
                inlier_count = int(mask.sum())

                if inlier_count >= MIN_GOOD_MATCHES:
                    # Project original baseline ring center into current live frame
                    baseline_center_np = np.float32([
                        [BASELINE_RING_CENTER]
                    ])

                    projected_center = cv2.perspectiveTransform(
                        baseline_center_np,
                        H
                    )

                    tx, ty = projected_center[0][0]
                    target_ring_center = (int(tx), int(ty))

                    match_status = f"Matches: {len(good_matches)} | Inliers: {inlier_count}"
                else:
                    match_status = f"Too few RANSAC inliers: {inlier_count}"
            else:
                match_status = "Homography failed"
        else:
            match_status = f"Too few good matches: {len(good_matches)}"
    else:
        match_status = "No live descriptors"

    # -----------------------------
    # Draw output
    # -----------------------------
    display = ring_debug_frame.copy()

    cv2.putText(
        display,
        match_status,
        (30, display.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if target_ring_center is not None:
        # Draw target ring position from baseline
        cv2.circle(display, target_ring_center, BASELINE_RING_RADIUS, (0, 255, 0), 2)
        cv2.circle(display, target_ring_center, 8, (0, 255, 0), -1)

        if current_ring_center is not None:
            display, error = draw_guidance(
                display,
                current_ring_center,
                target_ring_center
            )
        else:
            cv2.putText(
                display,
                "Current ring not detected",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )
    else:
        cv2.putText(
            display,
            "Target ring position not found",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    cv2.imshow("Live Metallic Ring Guide", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        save_path = OUTPUT_DIR / f"saved_frame_{frame_count}.jpg"
        cv2.imwrite(str(save_path), display)
        print(f"Saved {save_path}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
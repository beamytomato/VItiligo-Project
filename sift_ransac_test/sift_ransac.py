from pathlib import Path
import cv2
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASELINE_PATH = Path("images/baseline.jpg")
FOLLOWUP_PATH = Path("images/followup.jpg")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MATCH_OUTPUT = OUTPUT_DIR / "ransac_matches.jpg"

# -----------------------------
# Load images
# -----------------------------
baseline = cv2.imread(str(BASELINE_PATH))
followup = cv2.imread(str(FOLLOWUP_PATH))

if baseline is None:
    raise FileNotFoundError(f"Could not load {BASELINE_PATH}")

if followup is None:
    raise FileNotFoundError(f"Could not load {FOLLOWUP_PATH}")

# Convert to grayscale
baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
followup_gray = cv2.cvtColor(followup, cv2.COLOR_BGR2GRAY)

# Optional contrast improvement
baseline_gray = cv2.equalizeHist(baseline_gray)
followup_gray = cv2.equalizeHist(followup_gray)

# -----------------------------
# Step 1: Detect SIFT keypoints
# -----------------------------
sift = cv2.SIFT_create(nfeatures=3000)

kp1, des1 = sift.detectAndCompute(baseline_gray, None)
kp2, des2 = sift.detectAndCompute(followup_gray, None)

print(f"Baseline keypoints: {len(kp1)}")
print(f"Follow-up keypoints: {len(kp2)}")

if des1 is None or des2 is None:
    raise ValueError("No descriptors found. Try sharper images or better lighting.")

# -----------------------------
# Step 2: Match SIFT descriptors
# -----------------------------
# SIFT descriptors are floating point, so FLANN is commonly used.
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# For each descriptor in baseline, find 2 best matches in follow-up
matches = flann.knnMatch(des1, des2, k=2)

# -----------------------------
# Step 3: Lowe's ratio test
# -----------------------------
# This removes weak/ambiguous matches.
good_matches = []

for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"Good matches after ratio test: {len(good_matches)}")

MIN_MATCH_COUNT = 10

if len(good_matches) < MIN_MATCH_COUNT:
    raise ValueError(
        f"Not enough good matches: {len(good_matches)}. "
        "Try using a larger skin area, better lighting, or less camera movement."
    )

# -----------------------------
# Step 4: Prepare matched point coordinates
# -----------------------------
# src_pts = points in baseline image
# dst_pts = matching points in follow-up image
src_pts = np.float32([
    kp1[m.queryIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

dst_pts = np.float32([
    kp2[m.trainIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

# -----------------------------
# Step 5: RANSAC homography
# -----------------------------
# H maps baseline image coordinates to follow-up image coordinates.
# RANSAC rejects bad matches.
H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise ValueError("RANSAC could not find a valid homography.")

matches_mask = mask.ravel().tolist()
inlier_count = sum(matches_mask)

print(f"RANSAC inliers: {inlier_count} / {len(good_matches)}")
print("Homography matrix:")
print(H)

# -----------------------------
# Step 6: Draw only RANSAC inlier matches
# -----------------------------
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matches_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

ransac_match_image = cv2.drawMatches(
    baseline,
    kp1,
    followup,
    kp2,
    good_matches,
    None,
    **draw_params
)

cv2.imwrite(str(MATCH_OUTPUT), ransac_match_image)

print(f"Saved RANSAC match image to: {MATCH_OUTPUT}")
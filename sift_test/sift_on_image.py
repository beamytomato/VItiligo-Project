from pathlib import Path
import cv2

# -----------------------------
# Paths
# -----------------------------
IMAGE_PATH = Path("images/skin_ring.jpg")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_KEYPOINTS = OUTPUT_DIR / "sift_keypoints.jpg"

# -----------------------------
# Load image
# -----------------------------
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Convert to grayscale because SIFT works on image intensity
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: improve contrast
gray = cv2.equalizeHist(gray)

# -----------------------------
# Create SIFT detector
# -----------------------------
sift = cv2.SIFT_create(nfeatures=1000)

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

print(f"Number of SIFT keypoints found: {len(keypoints)}")

if descriptors is not None:
    print(f"Descriptor shape: {descriptors.shape}")
else:
    print("No descriptors found.")

# -----------------------------
# Draw keypoints on image
# -----------------------------
# -----------------------------
# Draw keypoints on image more visibly
# -----------------------------
keypoint_image = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Save output
cv2.imwrite(str(OUTPUT_KEYPOINTS), keypoint_image)

print(f"Saved SIFT keypoint image to: {OUTPUT_KEYPOINTS}")

# Optional: show image in a popup window
cv2.imshow("SIFT Keypoints", keypoint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
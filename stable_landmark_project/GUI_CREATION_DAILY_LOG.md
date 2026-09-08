# Stable Landmark Labeler GUI Daily Log

Date: 2026-06-24  
Project: `/Users/jonathanrhee/Projects/Sift_vitiligo_Opencv`  
Main model script: `stable_landmark_project/stable_landmark_labeler_v2.py`  
GUI entry point: `stable_landmark_project/stable_landmark_gui.py`

## Daily Work Log

### 2026-06-24

Created a local browser-based graphical interface for the automatic corresponding stable skin landmark labeler.

Work completed:

- Checked whether Streamlit or common local web frameworks were installed.
- Found that Streamlit, Flask, FastAPI, and Uvicorn were not installed in the local environment.
- Built the GUI using only Python standard library web server tools so no extra package install is required.
- Kept the computer vision model intact by calling `stable_landmark_labeler_v2.py` through `subprocess`.
- Added a browser form for uploading a baseline image and follow-up image.
- Added common model controls for landmark count, RootSIFT, cross-check matching, ring detection, debug outputs, polygon ROI mode, RANSAC threshold, minimum inliers, and minimum inlier ratio.
- Added post-run result display for pass/fail status, safety warnings, reliability metrics, run log, and generated output images.
- Added tabbed output image review for labeled landmarks, masks, keypoints, all good matches, and RANSAC inlier matches.
- Added `README_GUI.md` with launch instructions.
- Tested the labeler with sample images `stable_landmark_project/images/visit1good.jpg` and `stable_landmark_project/images/visit2good.jpg`.
- Verified that the browser form route could trigger the labeler and produce the expected output files.

## What The GUI Does

The GUI is a local browser interface for `stable_landmark_labeler_v2.py`. It is designed for clinical/research review of longitudinal vitiligo/FLAME image pairs.

The GUI does not replace or rewrite the landmark matching model. Instead, it:

1. Accepts an Image 1 / baseline upload.
2. Accepts an Image 2 / follow-up upload.
3. Saves the uploaded files into a per-run folder.
4. Builds a command-line call to `stable_landmark_labeler_v2.py`.
5. Runs that command with `subprocess`.
6. Reads `landmark_matches.json`.
7. Displays reliability metrics and output images in the browser.

The model remains conservative. If the labeler reports failure or says the alignment should not be trusted, the GUI presents that warning clearly rather than hiding it.

## How To Launch The GUI

From the project root:

```bash
cd /Users/jonathanrhee/Projects/Sift_vitiligo_Opencv
python3 stable_landmark_project/stable_landmark_gui.py
```

Then open:

```text
http://127.0.0.1:8501
```

If port `8501` is already in use:

```bash
python3 stable_landmark_project/stable_landmark_gui.py --port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

To stop the GUI server, go to the terminal where it is running and press `Control+C`.

## What The GUI Should Output

Each run writes files to the output directory selected in the form. If the output directory is left blank, the GUI creates a folder under:

```text
stable_landmark_project/gui_runs/<run_id>/outputs/
```

The uploaded source images are saved under:

```text
stable_landmark_project/gui_runs/<run_id>/uploads/
```

Expected model output files include:

- `labeled_landmarks.jpg`
- `landmark_matches.csv`
- `landmark_matches.json`
- `image1_allowed_mask.png`
- `image2_allowed_mask.png`
- `image1_keypoints.jpg`
- `image2_keypoints.jpg`
- `all_good_matches_before_ransac.jpg`
- `ransac_inlier_matches.jpg`

Some runs may also produce optional debug images such as mole candidate visualizations, depending on which model paths are used.

## What The Browser Displays After A Run

The GUI displays:

- Pass/fail status from `landmark_matches.json`.
- The model warning message if the alignment should not be trusted.
- Output directory path.
- Runtime and command log.
- Reliability metrics.
- Image tabs for reviewing visual artifacts.

The reliability metrics shown include:

- Keypoints image 1
- Keypoints image 2
- Raw matches
- Good matches after Lowe ratio
- RANSAC inliers
- Inlier ratio
- Average reprojection error
- Geometric model
- RootSIFT enabled/disabled
- Cross-check matches enabled/disabled
- Homography method, such as `USAC_MAGSAC` or fallback `RANSAC`

## GUI Settings

### Image 1 / Baseline

Upload the baseline image from the first visit.

### Image 2 / Follow-Up

Upload the follow-up image from the later visit.

### Output Directory

Optional local folder where outputs should be saved. If blank, the GUI auto-generates a run folder under `stable_landmark_project/gui_runs/`.

### Top Landmarks

Controls `--top`. This is the maximum number of final landmarks the model should label.

### RANSAC Threshold

Controls `--ransac-threshold`. This is the reprojection error threshold, in pixels, used during geometric model fitting.

### Min Inliers

Controls `--min-inliers`. This is the minimum number of RANSAC inlier matches required for the model to trust the geometry unless another accepted support path applies.

### Min Inlier Ratio

Controls `--min-inlier-ratio`. This is the minimum fraction of good matches that must survive RANSAC.

### RootSIFT Descriptors

Controls `--use-rootsift` or `--no-use-rootsift`. RootSIFT is enabled by default and improves histogram-style descriptor matching.

### Mutual Nearest-Neighbor Cross-Check

Controls `--cross-check-matches` or `--no-cross-check-matches`. This is enabled by default and keeps only matches that agree in both matching directions.

### Auto-Detect Baseline Ring

Controls `--auto-detect-baseline-ring`. When enabled, the labeler tries to find a circular measurement ring in the baseline image and suppress its metal annulus from the allowed landmark region.

### Save Debug Outputs

Controls `--save-debug`. When enabled, the labeler writes masks, keypoints, all-good matches, and RANSAC inlier images.

### Select Polygon ROI

Controls `--select-polygon-roi`. This is available but advanced because it opens OpenCV desktop windows while the browser waits. It is useful when a user wants to manually restrict the skin region.

## Output Image Tabs

### Final Labeled Output

Shows `labeled_landmarks.jpg`, the side-by-side baseline/follow-up image with corresponding landmarks numbered and boxed.

### Allowed Masks

Shows `image1_allowed_mask.png` and `image2_allowed_mask.png`, which reveal where the automatic quality/skin mask allowed landmark detection.

### Keypoints

Shows `image1_keypoints.jpg` and `image2_keypoints.jpg`, which visualize detected SIFT/stable spot keypoints inside the allowed masks.

### All Good Matches

Shows `all_good_matches_before_ransac.jpg`, the descriptor matches that passed Lowe ratio and gating before final geometric filtering.

### RANSAC Inlier Matches

Shows `ransac_inlier_matches.jpg`, the matches that survived geometric consistency checks.

## How The GUI Was Created

The GUI was implemented in `stable_landmark_gui.py`.

Implementation approach:

- Used `ThreadingHTTPServer` and `BaseHTTPRequestHandler` from Python standard library.
- Used `cgi.FieldStorage` to parse multipart browser uploads.
- Used `subprocess.run(...)` to call `stable_landmark_labeler_v2.py` with command-line options.
- Used `json` to read `landmark_matches.json` after the model finishes.
- Used a small in-memory run map so the browser can request generated artifact images.
- Used restrained HTML/CSS for a clinical/research layout with panels, metric cards, warning states, and tabs.
- Avoided adding new dependencies because Streamlit and common Python web frameworks were not installed locally.
- Avoided rewriting the computer vision model so the GUI stays a wrapper around the trusted existing script.

## Safety And Interpretation

The GUI should not be treated as proof that the image alignment is correct. It is a review surface for the existing conservative model.

If the status is failed, or if the message says not to trust the alignment, the user should not use the landmarks as reliable correspondences. In that case, try tighter image overlap, better lighting, a polygon ROI, or images with more stable visible skin marks.

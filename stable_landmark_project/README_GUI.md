# Stable Landmark Labeler GUI

This is a local browser interface for `stable_landmark_labeler_v2.py`. It does not replace the landmark model; it uploads the selected image pair, calls the existing labeler through `subprocess`, then displays the JSON reliability metrics and output images.

## Launch

From the project root:

```bash
python3 stable_landmark_project/stable_landmark_gui.py
```

Then open:

```text
http://127.0.0.1:8501
```

You can choose another port if needed:

```bash
python3 stable_landmark_project/stable_landmark_gui.py --port 8502
```

## Notes

- Leave the output directory blank to auto-create a run folder under `stable_landmark_project/gui_runs/`.
- The polygon ROI option is available, but it opens OpenCV desktop windows while the browser waits for the run to finish.
- The GUI shows a warning whenever the labeler reports a failed or untrusted alignment.
- Streamlit is not installed in this environment, so this GUI uses only the Python standard library and the existing OpenCV/Pillow stack.

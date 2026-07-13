"""
Browser GUI for stable_landmark_labeler_v2.py.

Launch:
    python3 stable_landmark_gui.py
"""

from __future__ import annotations

import argparse
import contextlib
from email import policy
from email.parser import BytesParser
import html
import importlib
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse


IS_FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_DIR = (
    Path(getattr(sys, "_MEIPASS"))
    if IS_FROZEN and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent
)
LABELER_SCRIPT = PROJECT_DIR / "stable_landmark_labeler_v2.py"
KNOWN_LANDMARKS_FILE = PROJECT_DIR / "patient461_known_landmarks.json"
DEFAULT_RUNS_DIR = PROJECT_DIR / "gui_runs"
RUNS: Dict[str, Path] = {}


class UploadedFile:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = io.BytesIO(data)

FAIL_WARNING = "Do not trust this alignment"
ARTIFACTS = {
    "Final labeled output": ["labeled_landmarks.jpg"],
    "Allowed masks": ["image1_allowed_mask.png", "image2_allowed_mask.png"],
    "Keypoints": ["image1_keypoints.jpg", "image2_keypoints.jpg"],
    "All good matches": ["all_good_matches_before_ransac.jpg"],
    "RANSAC inlier matches": ["ransac_inlier_matches.jpg"],
}
METRIC_KEYS = [
    ("Keypoints image 1", "total_keypoints_image1"),
    ("Keypoints image 2", "total_keypoints_image2"),
    ("Raw matches", "raw_matches"),
    ("Good matches after Lowe ratio", "good_matches_after_lowe"),
    ("RANSAC inliers", "ransac_inliers"),
    ("Inlier ratio", "inlier_ratio"),
    ("Average reprojection error", "average_reprojection_error"),
    ("Geometric model", "geometric_model"),
    ("RootSIFT", "use_rootsift"),
    ("Cross-check matches", "cross_check_matches"),
    ("Homography method", "homography_method"),
]


CSS = """
:root {
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #607080;
  --line: #d8e0e7;
  --blue: #1f5f8b;
  --blue-dark: #16476a;
  --green-bg: #e8f5ef;
  --green: #16603f;
  --red-bg: #fff0f0;
  --red: #9d2525;
  --amber-bg: #fff7e6;
  --amber: #855900;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
}
header {
  border-bottom: 1px solid var(--line);
  background: #ffffff;
  padding: 18px 28px;
}
h1 {
  margin: 0;
  font-size: 23px;
  font-weight: 700;
  letter-spacing: 0;
}
.subtitle {
  margin-top: 5px;
  color: var(--muted);
}
main {
  display: grid;
  grid-template-columns: minmax(330px, 430px) minmax(0, 1fr);
  gap: 18px;
  padding: 18px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
}
.stack { display: grid; gap: 14px; }
h2 {
  margin: 0 0 14px;
  font-size: 15px;
  font-weight: 700;
  color: #24313d;
}
label {
  display: block;
  color: #2e3b46;
  font-weight: 600;
  margin-bottom: 6px;
}
input[type="file"], input[type="text"], input[type="number"] {
  width: 100%;
  border: 1px solid #c8d3dc;
  border-radius: 6px;
  padding: 9px 10px;
  background: #fff;
  color: var(--ink);
}
input[type="checkbox"] {
  width: 17px;
  height: 17px;
  margin: 0;
}
.check {
  display: grid;
  grid-template-columns: 22px 1fr;
  align-items: start;
  gap: 8px;
  color: #273642;
}
.check span { line-height: 1.35; }
.hint {
  color: var(--muted);
  font-size: 13px;
  margin-top: 5px;
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
button {
  width: 100%;
  border: 0;
  border-radius: 6px;
  padding: 11px 14px;
  background: var(--blue);
  color: #fff;
  font-weight: 700;
  cursor: pointer;
}
button:hover { background: var(--blue-dark); }
.status {
  border-radius: 8px;
  padding: 14px 16px;
  border: 1px solid var(--line);
  background: #f8fafc;
}
.status.success {
  border-color: #b8decf;
  background: var(--green-bg);
  color: var(--green);
}
.status.failed {
  border-color: #f0c7c7;
  background: var(--red-bg);
  color: var(--red);
}
.status.warning {
  border-color: #efd48a;
  background: var(--amber-bg);
  color: var(--amber);
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfd;
}
.metric .name {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.metric .value {
  margin-top: 5px;
  font-size: 20px;
  font-weight: 700;
  word-break: break-word;
}
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}
.tab {
  width: auto;
  background: #e8eef3;
  color: #24313d;
  padding: 8px 11px;
  font-weight: 700;
}
.tab.active {
  background: var(--blue);
  color: #fff;
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.artifact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 14px;
}
.artifact {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfcfd;
}
.artifact img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  border: 1px solid #e0e6eb;
}
.artifact-title {
  color: #33414d;
  font-weight: 700;
  margin-bottom: 8px;
}
pre {
  white-space: pre-wrap;
  overflow: auto;
  max-height: 360px;
  border-radius: 8px;
  background: #111827;
  color: #dbeafe;
  padding: 14px;
  font-size: 13px;
}
.empty {
  color: var(--muted);
  border: 1px dashed #c7d2dc;
  border-radius: 8px;
  padding: 22px;
  text-align: center;
  background: #fbfcfd;
}
@media (max-width: 980px) {
  main { grid-template-columns: 1fr; }
}
"""


JS = """
function showTab(id) {
  document.querySelectorAll('.tab').forEach(button => button.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
  document.querySelector('[data-tab-button="' + id + '"]').classList.add('active');
  document.querySelector('[data-tab-panel="' + id + '"]').classList.add('active');
}
"""


def checked(fields: Dict[str, Any], name: str) -> bool:
    return name in fields


def text_value(fields: Dict[str, Any], name: str, default: str = "") -> str:
    item = fields.get(name)
    if item is None:
        return default
    if isinstance(item, list):
        item = item[0]
    value = getattr(item, "value", item)
    return str(value).strip()


def parse_multipart_form(headers: Any, body: bytes) -> Dict[str, Any]:
    content_type = headers.get("Content-Type", "")
    message_bytes = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    fields: Dict[str, Any] = {}

    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = UploadedFile(filename, payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")

    return fields


def safe_upload_basename(filename: str, default_stem: str) -> str:
    original = Path(filename or "").name
    suffix = Path(original).suffix or ".jpg"
    stem = Path(original).stem
    stem = re.sub(r"[^A-Za-z0-9._() -]+", "_", stem).strip(" ._")
    if not stem:
        stem = default_stem
    return f"{stem}{suffix}"


def save_upload(fields: Dict[str, Any], name: str, destination: Path) -> Optional[Path]:
    item = fields.get(name)
    if item is None or not getattr(item, "filename", ""):
        return None
    target_dir = destination / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_upload_basename(item.filename, name)
    with open(target, "wb") as f:
        shutil.copyfileobj(item.file, f)
    return target


def safe_output_dir(raw: str, run_id: str) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path
    return DEFAULT_RUNS_DIR / run_id / "outputs"


def run_labeler_command(cmd: List[str]) -> Any:
    if not IS_FROZEN:
        return subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    old_argv = sys.argv[:]
    output = io.StringIO()
    return_code = 0
    labeler_args = cmd[2:]

    try:
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))
        labeler = importlib.import_module("stable_landmark_labeler_v2")
        sys.argv = [str(LABELER_SCRIPT), *labeler_args]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            return_code = int(labeler.main() or 0)
    except SystemExit as exc:
        return_code = int(exc.code or 0) if isinstance(exc.code, int) else 1
    except Exception:
        import traceback

        return_code = 1
        output.write(traceback.format_exc())
    finally:
        sys.argv = old_argv

    return subprocess.CompletedProcess(cmd, return_code, output.getvalue(), None)


def run_labeler(fields: Dict[str, Any]) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    run_root = DEFAULT_RUNS_DIR / run_id
    upload_dir = run_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    image1 = save_upload(fields, "image1", upload_dir)
    image2 = save_upload(fields, "image2", upload_dir)
    if image1 is None or image2 is None:
        raise ValueError("Both baseline and follow-up images are required.")

    output_dir = safe_output_dir(text_value(fields, "output_dir"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    RUNS[run_id] = output_dir

    cmd = [
        sys.executable,
        str(LABELER_SCRIPT),
        "--image1",
        str(image1),
        "--image2",
        str(image2),
        "--output-dir",
        str(output_dir),
        "--top",
        text_value(fields, "top", "6"),
        "--ransac-threshold",
        text_value(fields, "ransac_threshold", "4.0"),
        "--min-inliers",
        text_value(fields, "min_inliers", "6"),
        "--min-inlier-ratio",
        text_value(fields, "min_inlier_ratio", "0.05"),
    ]
    if KNOWN_LANDMARKS_FILE.exists():
        cmd.extend(["--known-landmarks-file", str(KNOWN_LANDMARKS_FILE)])

    if checked(fields, "use_rootsift"):
        cmd.append("--use-rootsift")
    else:
        cmd.append("--no-use-rootsift")
    if checked(fields, "cross_check_matches"):
        cmd.append("--cross-check-matches")
    else:
        cmd.append("--no-cross-check-matches")
    if checked(fields, "auto_detect_baseline_ring"):
        cmd.append("--auto-detect-baseline-ring")
    if checked(fields, "save_debug"):
        cmd.append("--save-debug")
    if checked(fields, "select_polygon_roi"):
        cmd.append("--select-polygon-roi")

    started = time.time()
    completed = run_labeler_command(cmd)
    elapsed = time.time() - started

    payload: Dict[str, Any] = {}
    json_path = output_dir / "landmark_matches.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "elapsed": elapsed,
        "payload": payload,
    }


def page_shell(body: str) -> bytes:
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stable Landmark Labeler</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <h1>Stable Landmark Labeler</h1>
    <div class="subtitle">Conservative corresponding landmark matching for longitudinal vitiligo / FLAME clinical images</div>
  </header>
  {body}
  <script>{JS}</script>
</body>
</html>"""
    return doc.encode("utf-8")


def form_html(result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> str:
    return f"""
<main>
  <section class="stack">
    <form class="panel stack" method="post" action="/run" enctype="multipart/form-data">
      <div>
        <h2>Inputs</h2>
        <label for="image1">Image 1 / baseline</label>
        <input id="image1" name="image1" type="file" accept="image/*" required>
      </div>
      <div>
        <label for="image2">Image 2 / follow-up</label>
        <input id="image2" name="image2" type="file" accept="image/*" required>
      </div>
      <div>
        <label for="output_dir">Output directory</label>
        <input id="output_dir" name="output_dir" type="text" placeholder="Leave blank to auto-generate under stable_landmark_project/gui_runs">
        <div class="hint">Browser security does not expose a native folder picker; enter a local path or leave blank.</div>
      </div>
      <div class="grid-2">
        <div>
          <label for="top">Top landmarks</label>
          <input id="top" name="top" type="number" min="1" max="40" value="6">
        </div>
        <div>
          <label for="ransac_threshold">RANSAC threshold</label>
          <input id="ransac_threshold" name="ransac_threshold" type="number" min="0.5" max="30" step="0.5" value="4.0">
        </div>
        <div>
          <label for="min_inliers">Min inliers</label>
          <input id="min_inliers" name="min_inliers" type="number" min="1" max="80" value="6">
        </div>
        <div>
          <label for="min_inlier_ratio">Min inlier ratio</label>
          <input id="min_inlier_ratio" name="min_inlier_ratio" type="number" min="0" max="1" step="0.01" value="0.05">
        </div>
      </div>
      <div>
        <h2>Common Options</h2>
        {checkbox("use_rootsift", "RootSIFT descriptors", True)}
        {checkbox("cross_check_matches", "Mutual nearest-neighbor cross-check", True)}
        {checkbox("auto_detect_baseline_ring", "Auto-detect baseline ring", False)}
        {checkbox("save_debug", "Save debug outputs", False)}
        {checkbox("select_polygon_roi", "Select polygon ROI using OpenCV windows", False, "Advanced: this opens desktop OpenCV selection windows during the run.")}
      </div>
      <button type="submit">Run Landmark Matching</button>
    </form>
  </section>
  <section class="stack">
    {render_error(error) if error else ""}
    {render_result(result) if result else render_empty()}
  </section>
</main>
"""


def checkbox(name: str, label: str, default: bool, hint: str = "") -> str:
    checked_attr = " checked" if default else ""
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return f"""
<label class="check">
  <input type="checkbox" name="{html.escape(name)}"{checked_attr}>
  <span>{html.escape(label)}{hint_html}</span>
</label>
"""


def render_error(error: str) -> str:
    return f'<div class="status failed"><strong>Unable to run:</strong> {html.escape(error)}</div>'


def render_empty() -> str:
    return """
<div class="panel">
  <div class="empty">Run a pair of images to review generated output artifacts.</div>
</div>
"""


def value_display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_result(result: Dict[str, Any]) -> str:
    payload = result.get("payload") or {}
    status = str(payload.get("status") or "unknown")
    message = str(payload.get("message") or "")
    output_dir = Path(result["output_dir"])
    warning = status.lower() != "success" or FAIL_WARNING.lower() in message.lower()
    status_class = "success" if status.lower() == "success" else "failed"

    warning_html = ""
    if warning:
        warning_text = message or "The labeler did not report a reliable success status."
        warning_html = f'<div class="status warning"><strong>Review required:</strong> {html.escape(warning_text)}</div>'

    tabs_html = render_artifact_tabs(result["run_id"], output_dir)
    command = " ".join(str(part) for part in result["command"])
    return f"""
<div class="panel stack">
  <div class="status {status_class}">
    <strong>Status:</strong> {html.escape(status.upper())}
    <span class="hint">Return code {result["returncode"]} · {result["elapsed"]:.1f}s · {html.escape(str(output_dir))}</span>
  </div>
  {warning_html}
</div>
<div class="panel">
  <h2>Output Images</h2>
  {tabs_html}
</div>
<div class="panel">
  <h2>Run Log</h2>
  <div class="hint">Command: {html.escape(command)}</div>
  <pre>{html.escape(result["stdout"] or "")}</pre>
</div>
"""


def render_artifact_tabs(run_id: str, output_dir: Path) -> str:
    buttons: List[str] = []
    panels: List[str] = []
    first = True
    for label, filenames in ARTIFACTS.items():
        tab_id = label.lower().replace(" ", "-")
        active = " active" if first else ""
        buttons.append(
            f'<button class="tab{active}" type="button" data-tab-button="{tab_id}" onclick="showTab(\'{tab_id}\')">{html.escape(label)}</button>'
        )
        artifact_items = []
        for filename in filenames:
            path = output_dir / filename
            if path.exists():
                src = f"/artifact?run={html.escape(run_id)}&file={html.escape(filename)}"
                artifact_items.append(
                    f'<div class="artifact"><div class="artifact-title">{html.escape(filename)}</div><img src="{src}" alt="{html.escape(filename)}"></div>'
                )
            else:
                artifact_items.append(
                    f'<div class="artifact"><div class="artifact-title">{html.escape(filename)}</div><div class="empty">Not produced for this run.</div></div>'
                )
        panels.append(
            f'<div class="tab-panel{active}" data-tab-panel="{tab_id}"><div class="artifact-grid">{"".join(artifact_items)}</div></div>'
        )
        first = False
    return f'<div class="tabs">{"".join(buttons)}</div>{"".join(panels)}'


class LandmarkGuiHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(form_html())
        elif parsed.path == "/artifact":
            self.send_artifact(parsed.query)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/run":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            fields = parse_multipart_form(self.headers, self.rfile.read(content_length))
            result = run_labeler(fields)
            self.send_html(form_html(result=result))
        except Exception as exc:
            self.send_html(form_html(error=str(exc)), status=400)

    def send_html(self, body: str, status: int = 200) -> None:
        data = page_shell(body)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_artifact(self, query: str) -> None:
        params = parse_qs(query)
        run_id = params.get("run", [""])[0]
        filename = params.get("file", [""])[0]
        output_dir = RUNS.get(run_id)
        allowed_files = {name for filenames in ARTIFACTS.values() for name in filenames}
        if output_dir is None or filename not in allowed_files:
            self.send_error(404)
            return
        path = output_dir / filename
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[stable-landmark-gui] {self.address_string()} - {format % args}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local browser GUI for stable_landmark_labeler_v2.py")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("PORT", "8501")), type=int)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=IS_FROZEN)
    args = parser.parse_args(list(argv) if argv is not None else None)

    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), LandmarkGuiHandler)
    print(f"Stable Landmark Labeler GUI running at http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")
    if args.open_browser:
        browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{browser_host}:{args.port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

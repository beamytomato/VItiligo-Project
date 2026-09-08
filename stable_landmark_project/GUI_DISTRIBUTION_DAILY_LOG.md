# Stable Landmark Labeler GUI Distribution Daily Log

Project: `/Users/jonathanrhee/Projects/Sift_vitiligo_Opencv`  
GUI entry point: `stable_landmark_project/stable_landmark_gui.py`  
Model script: `stable_landmark_project/stable_landmark_labeler_v2.py`  
Related notes:

- `stable_landmark_project/GUI_CREATION_DAILY_LOG.md`
- `stable_landmark_project/README_GUI.md`
- `stable_landmark_project/CLOUD_RUN_DEPLOYMENT.md`

## Overall Goal

After creating the browser-based GUI for the stable landmark labeler, the next goal was to figure out how another person could use it without depending on the original developer's laptop being on.

The practical question was:

```text
How can someone else access and use the GUI whenever they need it?
```

We explored three main approaches:

1. Share the local server over the same Wi-Fi.
2. Publish/deploy the GUI online through a cloud platform.
3. Package the GUI as a local Mac application using PyInstaller.

The final practical direction chosen was PyInstaller for local distribution, because it lets another Mac user run the GUI on their own computer without cloud billing or a server that must stay on.

## Daily Log

### 2026-07-02: Brainstormed How To Make The GUI Accessible

#### Goal

Understand how another person could use the GUI from a different device.

#### Key Question

The GUI was running locally at:

```text
http://127.0.0.1:8501
```

The user wanted to know whether another person could access that interface without the developer manually running the Python server.

#### What We Clarified

`127.0.0.1` only points to the current computer. If the GUI is running on one Mac, another device cannot use `127.0.0.1` to reach it.

For same-Wi-Fi sharing, the GUI can be run with:

```bash
python3 stable_landmark_project/stable_landmark_gui.py --host 0.0.0.0 --port 8501
```

Then another device on the same network could visit:

```text
http://<host-mac-local-ip>:8501
```

#### Challenge

This still requires the host Mac to be:

- awake
- connected to the network
- running the Python GUI server

If the host Mac is off, the website is unavailable.

#### Solution/Decision

We clarified the core rule:

```text
A web app must run somewhere.
```

If no local computer is running the Python server, the app must either:

- run on the other person's computer
- run on an always-on lab/server machine
- run on a cloud hosting platform

### 2026-07-02: Discussed Static Website Publishing

#### Goal

Determine whether the GUI could be published like a normal static website.

#### What We Clarified

The GUI is not just static HTML/CSS/JavaScript. It performs server-side work:

- accepts uploaded images
- saves files
- runs Python code
- imports OpenCV and NumPy
- runs SIFT/RootSIFT matching
- runs RANSAC/homography geometry
- creates CSV/JSON outputs
- writes debug images
- serves generated images back to the browser

Static hosting platforms can display files, but they cannot run:

```bash
python3 stable_landmark_project/stable_landmark_labeler_v2.py
```

#### Challenge

A static site host such as GitHub Pages is not enough for this app because the computer vision pipeline needs a backend runtime.

#### Solution/Decision

We separated the deployment categories:

```text
static website = display-only files
stable landmark GUI = upload + compute + generated outputs
```

Therefore, the app needs either:

- Python-capable hosting
- local execution on the user's machine
- a packaged desktop/local app

### 2026-07-02: Explored Google Cloud / Cloud Run

#### Goal

Try to deploy the GUI so someone else could use it from a URL while the original laptop is off.

#### Why Google Cloud Run Was Considered

Cloud Run is a good technical fit because:

- it can run Python apps
- it can run OpenCV inside a container
- it provides a public or restricted HTTPS URL
- it can scale down when unused
- the user's laptop does not need to stay on

The target architecture was:

```text
Browser
  -> Google Cloud Run URL
  -> stable_landmark_gui.py
  -> stable_landmark_labeler_v2.py
  -> generated outputs shown in browser
```

#### Project Preparation

We prepared the repository for Cloud Run by adding:

- `requirements.txt`
- `Dockerfile`
- `.dockerignore`
- `stable_landmark_project/CLOUD_RUN_DEPLOYMENT.md`

We also updated the GUI so it could read the Cloud Run port from the environment:

```python
os.environ.get("PORT", "8501")
```

This matters because Cloud Run provides its own port through the `PORT` environment variable.

#### Google Cloud CLI Setup

The user installed the Google Cloud CLI on macOS ARM64.

Important setup decisions:

- For the anonymized usage-data prompt, choosing `N` was acceptable for privacy.
- For shell profile modification, choosing `Y` was useful so the `gcloud` command would be available.
- When asked which rc file to update, pressing Enter selected the default `.zshrc`.
- When `gcloud` was not on the path yet, using the direct SDK path worked:

```bash
./bin/gcloud init
```

#### Project Creation Challenge

The first Google Cloud project ID was too long.

Google returned:

```text
project_id must be at most 30 characters long
project display name must be at most 30 characters
```

#### Solution

Use a shorter project ID, for example:

```bash
./bin/gcloud projects create sift-landmark-gui
./bin/gcloud config set project sift-landmark-gui
```

The shorter project ID worked.

#### Deployment Attempt

The intended deployment command was:

```bash
gcloud run deploy stable-landmark-labeler --source . --region us-west1
```

When `gcloud` was not on the shell path, the direct SDK path could be used:

```bash
~/Downloads/google-cloud-sdk/bin/gcloud run deploy stable-landmark-labeler --source . --region us-west1
```

#### Billing Challenge

Cloud Run deployment failed because billing was not enabled.

The error indicated:

```text
FAILED_PRECONDITION: Billing account for project is not found.
Billing must be enabled for activation of service(s)
artifactregistry.googleapis.com
cloudbuild.googleapis.com
run.googleapis.com
containerregistry.googleapis.com
```

#### What This Meant

Google Cloud required a billing account before enabling the APIs needed for Cloud Run deployment.

The charge-related services included:

- Cloud Build to build the app
- Artifact Registry to store the container image
- Cloud Run to serve the app
- networking/logging/storage depending on usage

#### Decision

We did not continue with Google Cloud as the chosen path because enabling billing meant the project could create costs.

The conclusion was:

```text
Cloud Run is technically correct, but not ideal if the goal is avoiding billing.
```

### 2026-07-02: Discussed Free / Lower-Cost Alternatives

#### Goal

Find options that would let the user avoid Google Cloud billing while still making the GUI accessible.

#### Options Discussed

##### Hugging Face Spaces

Potentially useful because it supports Python apps and Docker-style demos.

Pros:

- can host machine-learning-style demos
- can provide a URL
- can be easier than raw cloud infrastructure

Challenges:

- public/private access limitations
- free resources may sleep or be limited
- not ideal for sensitive clinical images without privacy controls

##### Streamlit Community Cloud

Potentially useful if the GUI were rewritten as a Streamlit app.

Pros:

- friendly for Python data/image apps
- easy sharing

Challenges:

- current GUI is not Streamlit
- would require rewriting or adapting the interface
- privacy concerns remain

##### Render / Railway / Similar Platforms

Potentially useful for Python web apps.

Challenges:

- free tiers can sleep or have limits
- may still require account setup or payment information
- OpenCV workloads may be heavy for free tiers

##### Static Hosting

Rejected because the GUI needs Python/OpenCV execution.

##### Local App Distribution

Strong option because another user can run the app on their own machine.

#### Decision

Because the other person could run software locally, we moved toward packaging the GUI as a local app.

### 2026-07-02 to 2026-07-05: Settled On PyInstaller

#### Goal

Allow another Mac user to run the GUI locally without needing:

- the original developer's laptop
- Google Cloud
- a paid server
- manual Python command-line setup every time

#### What PyInstaller Does

PyInstaller packages a Python program into a local application bundle.

For this project, the intended flow became:

```text
User double-clicks StableLandmarkGUI.app
  -> local web server starts on their Mac
  -> browser opens http://127.0.0.1:8501
  -> user uploads images
  -> their own Mac runs OpenCV/SIFT/RANSAC
  -> results are saved locally
```

#### Why This Solved The Original Problem

The other person can use the GUI whenever they want because their own computer becomes the server.

This avoids:

- keeping the original developer's computer on
- cloud billing
- public hosting risks
- uploading clinical images to a third-party server

#### Key Limitation

PyInstaller does not make a public website.

It creates a local app:

```text
PyInstaller = local app distribution
Cloud hosting = public or shared web URL
```

### 2026-07-05: Made The GUI More PyInstaller-Compatible

#### Goal

Make the existing GUI work better after being packaged into a Mac application.

#### Challenge

The GUI originally called the labeler with a subprocess command using:

```python
sys.executable
```

That works during normal development because `sys.executable` points to Python.

But inside a PyInstaller app, `sys.executable` points to the packaged app executable, not a normal Python interpreter. That could make subprocess-based labeler execution fail.

#### Solution

The GUI was adjusted so that when it is running as a frozen PyInstaller app, it calls the labeler inside the same process instead of trying to launch a separate Python interpreter.

Conceptually:

```text
normal development mode:
    GUI calls stable_landmark_labeler_v2.py through subprocess

PyInstaller frozen mode:
    GUI imports stable_landmark_labeler_v2 and runs main() internally
```

This kept normal local development behavior intact while making the packaged app more reliable.

#### Additional Improvement

The packaged app was also adjusted to open the browser automatically when double-clicked.

That improves user experience because the user does not need to manually type:

```text
http://127.0.0.1:8501
```

### 2026-07-05: PyInstaller Build Plan

#### Goal

Build a distributable Mac app for another Mac user.

#### Recommended Build Environment

Use a clean virtual environment so the package does not accidentally include unnecessary libraries.

Commands:

```bash
cd /Users/jonathanrhee/Projects/Sift_vitiligo_Opencv
python3 -m venv .venv-pyinstaller
source .venv-pyinstaller/bin/activate
pip install --upgrade pip
pip install pyinstaller numpy opencv-python
```

#### Build Command

Recommended PyInstaller command:

```bash
python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name StableLandmarkGUI \
  --hidden-import stable_landmark_labeler_v2 \
  --paths stable_landmark_project \
  stable_landmark_project/stable_landmark_gui.py
```

Expected output:

```text
dist/StableLandmarkGUI.app
```

#### Testing

The app should be tested locally by double-clicking:

```text
dist/StableLandmarkGUI.app
```

Expected behavior:

- local server starts
- browser opens automatically
- GUI appears at `http://127.0.0.1:8501`
- image upload and landmark matching work
- results save locally

### 2026-07-05: Sharing The App

#### Goal

Share the packaged Mac app with another person.

#### Challenge

The zipped app may be too large for normal messaging apps or Discord.

#### Solutions Discussed

Recommended sharing options:

- Google Drive
- Dropbox
- OneDrive
- iCloud Drive
- WeTransfer
- GitHub Releases

For GitHub, the recommendation was:

```text
Use GitHub Releases for the zip.
Do not commit the large zip directly into the repository.
```

Reason:

- large binary files make a git repository heavy
- every clone can pull binary history
- releases are intended for downloadable app builds

### 2026-07-05: macOS Gatekeeper Warning

#### Goal

Help the other Mac user open the packaged app.

#### Challenge

macOS showed a warning similar to:

```text
Apple could not verify that the app is free of malware.
```

#### Cause

The app was built locally and was not Apple-signed or notarized.

This is normal for unsigned PyInstaller apps downloaded from the internet.

#### Short-Term Solution

The other user can try:

```text
Right-click StableLandmarkGUI.app
Click Open
Click Open again
```

If blocked, they can remove the quarantine attribute:

```bash
xattr -dr com.apple.quarantine /path/to/StableLandmarkGUI.app
```

#### Long-Term Professional Solution

For a polished Mac app, the developer would need:

- Apple Developer account
- Developer ID certificate
- code signing
- notarization with Apple

That was not required for one trusted tester, but it would be needed for broader distribution.

### 2026-07-05: Reducing Zip Size

#### Goal

Make the app zip smaller and include only what is needed.

#### Challenge

PyInstaller packages can be large because they bundle:

- Python runtime
- NumPy
- OpenCV
- support libraries
- GUI script
- labeler script

The GUI code itself is small; most size comes from dependencies.

#### Solutions

Use a clean virtual environment:

```bash
python3 -m venv .venv-small-build
source .venv-small-build/bin/activate
pip install --upgrade pip
pip install pyinstaller numpy opencv-python
```

Avoid zipping the whole project. Only zip:

```text
dist/StableLandmarkGUI.app
```

Use `ditto` for macOS app zipping:

```bash
cd dist
ditto -c -k --sequesterRsrc --keepParent StableLandmarkGUI.app StableLandmarkGUI-mac.zip
```

Exclude unnecessary packages during build:

```bash
python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name StableLandmarkGUI \
  --hidden-import stable_landmark_labeler_v2 \
  --paths stable_landmark_project \
  --exclude-module matplotlib \
  --exclude-module pandas \
  --exclude-module scipy \
  --exclude-module sklearn \
  --exclude-module torch \
  --exclude-module tensorflow \
  stable_landmark_project/stable_landmark_gui.py
```

### 2026-07-06: Discussed Windows Compatibility

#### Goal

Understand what happens if the other person has a Windows computer.

#### Key Point

PyInstaller builds are platform-specific.

```text
Mac .app -> macOS only
Windows .exe -> Windows only
Linux build -> Linux only
```

#### Challenge

A Mac-built `.app` cannot run on Windows.

PyInstaller is not a simple cross-compiler. A Windows executable should usually be built on Windows.

#### Solution

If a Windows version is needed, build it on a Windows machine:

```powershell
cd path\to\Sift_vitiligo_Opencv
python -m venv .venv-build
.venv-build\Scripts\activate
pip install --upgrade pip
pip install pyinstaller numpy opencv-python
python -m PyInstaller --noconfirm --clean --windowed --name StableLandmarkGUI --hidden-import stable_landmark_labeler_v2 --paths stable_landmark_project stable_landmark_project\stable_landmark_gui.py
```

Expected Windows output:

```text
dist\StableLandmarkGUI\StableLandmarkGUI.exe
```

### 2026-07-06 to 2026-07-07: Chose GitHub Actions For The Windows Build

#### Goal

Create a Windows version of the Stable Landmark Labeler GUI even though the development computer was a Mac.

#### Challenge

The user only had access to a Mac computer. This created a packaging problem because PyInstaller builds are operating-system specific:

```text
Build on macOS  -> creates a macOS app
Build on Windows -> creates a Windows exe
```

A Windows `.exe` generally needs to be built in a Windows environment. Building it directly on the Mac would not reliably create a working Windows app.

#### Options Considered

We discussed several possible ways to create a Windows version:

- use a real Windows computer
- use a Windows virtual machine
- use a cloud Windows machine
- use GitHub Actions with a Windows runner

#### Solution

We chose GitHub Actions because it gives the project access to a temporary Windows machine in the cloud. The workflow checks out the repository, installs Python and dependencies, runs PyInstaller on Windows, zips the result, and provides it as a downloadable artifact.

This solved the main limitation:

```text
The developer can stay on Mac, while GitHub builds the Windows executable.
```

### 2026-07-07: Added A Windows PyInstaller GitHub Actions Workflow

#### Goal

Automate creation of a Windows zip package for the GUI.

#### Implemented File

We added:

```text
.github/workflows/build-windows-pyinstaller.yml
```

#### What The Workflow Does

The workflow runs manually through GitHub's **Actions** tab using `workflow_dispatch`.

The main steps are:

1. Check out the repository.
2. Set up Python `3.11` on a Windows runner.
3. Install build/runtime dependencies:

```powershell
python -m pip install --upgrade pip
pip install pyinstaller numpy opencv-python
```

4. Run PyInstaller:

```powershell
python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name StableLandmarkGUI `
  --hidden-import stable_landmark_labeler_v2 `
  --paths stable_landmark_project `
  stable_landmark_project\stable_landmark_gui.py
```

5. Zip the Windows app folder:

```powershell
Compress-Archive -Path dist\StableLandmarkGUI -DestinationPath StableLandmarkGUI-windows.zip -Force
```

6. Upload the zip as a GitHub Actions artifact:

```text
StableLandmarkGUI-windows
```

#### Why Python 3.11 Was Used

Python `3.11` was chosen because it is stable and well-supported by PyInstaller, NumPy, and OpenCV. This avoids some of the compatibility risk that can happen with very new Python versions.

#### Result

The project now had a repeatable way to produce a Windows package without needing a local Windows computer.

### 2026-07-07: Clarified How To Download And Share The Windows Build

#### Goal

Explain how the user could get the Windows app after GitHub Actions finished building it.

#### Workflow For The User

The intended process became:

1. Push the source code to GitHub.
2. Open the repository on GitHub.
3. Go to the **Actions** tab.
4. Select **Build Windows PyInstaller App**.
5. Click **Run workflow**.
6. Wait for the build to finish.
7. Open the completed workflow run.
8. Download the artifact named:

```text
StableLandmarkGUI-windows
```

9. Extract the downloaded zip on the Windows computer.
10. Run:

```text
StableLandmarkGUI.exe
```

#### Challenge

The downloadable file from GitHub Actions is an artifact zip. The user needs to fully extract it before running the `.exe`.

#### Solution

We clarified that the user should not run the executable directly from inside the compressed zip preview. The folder should be extracted first, then the `.exe` should be launched from the extracted folder.

### 2026-07-07: Encountered The Windows Python DLL Error

#### Goal

Diagnose why the Windows GUI failed to launch after packaging.

#### Error

The Windows computer showed an error similar to:

```text
Failed to load Python DLL
...\StableLandmarkGUI_internal\python311.dll
LoadLibrary: The specified module could not be found.
```

#### What The Error Meant

The executable was trying to load the bundled Python runtime, but Windows could not find or load the required Python DLL from the extracted package.

#### Most Likely Causes

The common causes were:

- the zip was not fully extracted
- the executable was being run from inside the compressed zip
- the internal PyInstaller folder was missing
- Windows security or antivirus quarantined a bundled file
- the package was moved/copied without all of its companion files

#### Solution

The recommended fix was to extract the entire zip into a normal folder first, then run the executable from that extracted folder.

The folder structure needs to stay intact. For a one-folder PyInstaller build, the `.exe` depends on nearby bundled files and internal folders.

Conceptually:

```text
StableLandmarkGUI/
  StableLandmarkGUI.exe
  _internal/
    python311.dll
    numpy/opencv/support files
```

If the `.exe` is separated from `_internal`, the app cannot start.

### 2026-07-07 to 2026-07-10: Improved The GUI For Packaged Builds

#### Goal

Make the same GUI source code work better in normal Python mode, macOS PyInstaller mode, and Windows PyInstaller mode.

#### Challenge

The GUI originally started the labeler by building a command around:

```python
sys.executable
```

In normal development, this points to a Python interpreter. In a PyInstaller app, it points to the packaged executable. If the GUI tried to call the packaged executable like it was Python, the labeler could fail.

#### Solution

The GUI was improved to detect when it is running as a frozen PyInstaller app:

```python
IS_FROZEN = bool(getattr(sys, "frozen", False))
```

When not frozen, the GUI still uses subprocess behavior.

When frozen, the GUI imports and runs the labeler inside the same packaged process. This makes the packaged app more reliable because it does not need to find a separate Python executable.

#### Why This Matters For Windows

This is especially useful for Windows because the user expects one packaged app folder to just run. The GUI should not depend on the user having Python installed separately or on the system path.

### 2026-07-10: Fixed Python 3.14 Compatibility For Local GUI Runs

#### Goal

Allow the GUI to run from the terminal after the local Python version changed.

#### Challenge

The GUI failed with:

```text
ModuleNotFoundError: No module named 'cgi'
```

#### Cause

The local computer was using a newer Python version where the old `cgi` module was no longer available.

#### Solution

The GUI multipart upload parsing was updated to avoid the removed `cgi` module. The replacement used the standard email parser:

```python
from email import policy
from email.parser import BytesParser
```

This kept the GUI dependency-light while making it compatible with newer Python versions.

#### Result

The GUI could again parse uploaded images without relying on the deprecated/removed `cgi` module.

### 2026-07-10: Handled Port Conflicts While Running The GUI

#### Goal

Help the user run the GUI from the terminal when the default port was already occupied.

#### Challenge

The terminal showed:

```text
OSError: [Errno 48] Address already in use
```

#### Cause

Another copy of the GUI or another local server was already using the selected port.

#### Solution

Use a different port:

```bash
python3 stable_landmark_project/stable_landmark_gui.py --port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

Alternatively, stop the older server with `Control+C` in the terminal where it was running.

### 2026-07-10: Confirmed OpenCV Dependency Requirements

#### Goal

Understand why a GUI run failed after image upload.

#### Challenge

The GUI itself loaded, but running the model failed with:

```text
ModuleNotFoundError: No module named 'cv2'
```

#### Cause

The Python interpreter used to run the GUI did not have OpenCV installed. The GUI is only the interface; the actual labeler still needs OpenCV for image processing and SIFT/RANSAC matching.

#### Solution

Install OpenCV in the same Python environment that runs the GUI:

```bash
python3 -m pip install opencv-python numpy
```

For PyInstaller builds, the GitHub Actions workflow installs:

```powershell
pip install pyinstaller numpy opencv-python
```

That ensures the Windows packaged app includes the necessary computer vision dependencies.

### 2026-07-10: Changed Save Debug Outputs Default

#### Goal

Make the Windows/shared GUI feel lighter for regular users.

#### Challenge

Debug outputs are useful during development, but they create many additional files:

- allowed masks
- keypoint images
- all good matches before RANSAC
- RANSAC inlier match views
- candidate/debug images

For a casual user, this can make output folders look cluttered.

#### Solution

The GUI default was changed so **Save debug outputs** starts unchecked.

#### Important Clarification

Turning debug outputs off does not mean the final result is lost.

The labeler should still save the core result files when the run succeeds:

```text
labeled_landmarks.jpg
landmark_matches.csv
landmark_matches.json
```

Debug mode only controls the extra diagnostic images.

### 2026-07-10: Clarified Output Directory Behavior On Windows

#### Goal

Explain how Windows users can choose where outputs are saved.

#### Challenge

The GUI runs in a browser, and browser security does not provide a native folder picker for arbitrary local save paths in this simple local app.

#### Solution

The user can type a Windows folder path into the output directory box, for example:

```text
C:\Users\Jonny\Desktop\Landmark Outputs
```

If the output directory field is left blank, the GUI auto-generates a run folder under:

```text
stable_landmark_project\gui_runs
```

In the packaged Windows app, the same idea applies: either enter a valid local path or let the app generate an output folder automatically.

### 2026-07-10: Created A Reusable Codex Prompt For Other Windows PyInstaller Projects

#### Goal

Make it easier to repeat the same Windows PyInstaller process for another project.

#### Challenge

The user wanted to do the same kind of Windows packaging for a different GUI project, but the exact files, dependencies, and entry point would be different.

#### Solution

We outlined the information Codex would need:

- project folder path
- GUI entry-point script
- main scripts/modules the GUI calls
- dependencies such as OpenCV, NumPy, Streamlit, Flask, etc.
- whether the app should be one-file or one-folder
- desired app name
- any data/model files that must be included
- target operating system
- whether GitHub Actions should build the app

This turned the experience from this project into a reusable checklist for packaging future Python GUIs.

## Final Decision

The final chosen distribution path was:

```text
Package the GUI with PyInstaller as a local app.
```

This was chosen because:

- users can run the GUI on their own Mac or Windows computer
- the app can run locally on their machine
- the original developer does not need to keep a server running
- no cloud billing is required
- patient/clinical images can stay on the user's own computer
- the app can be shared as a zip file
- GitHub Actions can build the Windows version even from a Mac-based development workflow

Google Cloud Run remained a valid technical option, but it was not chosen because billing was required.

## Remaining Improvements

Potential future improvements:

- Add Apple code signing and notarization to avoid Gatekeeper warnings.
- Add a small app icon.
- Add a better launch screen or status page.
- Add automatic cleanup of old GUI runs.
- Add a browser-based polygon ROI tool so OpenCV desktop windows are not needed.
- Continue testing the Windows build on real Windows machines.
- Consider adding a short `RUN_ME_FIRST.txt` inside the Windows zip explaining extraction and launch steps.
- Consider creating a GitHub Release for both Mac and Windows packages.

## Summary

The project started with a local browser GUI and explored how to make it available to another person.

We learned:

- same-Wi-Fi sharing works only while a host computer is running
- static website publishing is not enough because the app needs Python/OpenCV
- Google Cloud Run is technically appropriate but requires billing
- free hosting options exist but have privacy, sleep, or rewrite tradeoffs
- PyInstaller is the best fit for trusted users who can run the app locally
- Windows PyInstaller builds need a Windows environment
- GitHub Actions can provide that Windows environment without owning a Windows computer

The practical result was a local app distribution strategy:

```text
Build StableLandmarkGUI.app for Mac
Build StableLandmarkGUI.exe on Windows through GitHub Actions
Zip only the finished app package
Share through Drive/Release/large-file service
The other user extracts it and runs it locally whenever needed
```

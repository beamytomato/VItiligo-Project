# Deploying The Stable Landmark Labeler GUI To Google Cloud Run

This project can run on Google Cloud Run as a small Python/OpenCV web app.

## 1. Confirm Google Cloud CLI Is Ready

From any terminal:

```bash
gcloud config list
```

Confirm the correct account and project are selected.

If `gcloud` is still not on your shell path, run it from the SDK folder:

```bash
~/Downloads/google-cloud-sdk/bin/gcloud config list
```

## 2. Enable Billing

Cloud Run requires billing on the selected Google Cloud project.

Open:

```text
https://console.cloud.google.com/billing
```

Attach a billing account to the project before deploying.

## 3. Deploy From The Project Root

From this repository root:

```bash
cd /Users/jonathanrhee/Projects/Sift_vitiligo_Opencv
gcloud run deploy stable-landmark-labeler --source . --region us-west1
```

If `gcloud` is not on your path yet:

```bash
cd /Users/jonathanrhee/Projects/Sift_vitiligo_Opencv
~/Downloads/google-cloud-sdk/bin/gcloud run deploy stable-landmark-labeler --source . --region us-west1
```

During deployment, Google may ask whether to enable required APIs. Answer yes.

## 4. Access Choice

For an easy first test, Cloud Run may ask:

```text
Allow unauthenticated invocations?
```

Answering `Y` makes the app accessible to anyone with the URL.

For clinical or sensitive images, do not leave this public long-term. Add authentication or restrict access to approved Google accounts.

## 5. Use The URL

After deployment, Cloud Run prints a URL like:

```text
https://stable-landmark-labeler-xxxxx-uw.a.run.app
```

Open that URL in a browser.

## Notes

- The app now reads the `PORT` environment variable, which Cloud Run sets automatically.
- The Docker image uses `opencv-python-headless`, which is appropriate for a server without desktop windows.
- The polygon ROI option should not be used on Cloud Run because it opens OpenCV desktop windows.
- Cloud Run local disk is temporary. Uploaded images and outputs should be treated as temporary unless Google Cloud Storage is added later.

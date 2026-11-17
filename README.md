# Face Attendance Without Liveness

This version uses the same LBPH recognition and face cropping as the original project. Attendance is recorded after eight consecutive accepted matches for one student. There are no blink, head-turn, or eye calibration checks. Unknown faces, multiple detected faces and loss of a face reset the recognition count. Recognition thresholds still require calibration. This version can accept a photo or video of an enrolled student: it provides no anti-spoofing.

## Keep existing registrations

Extract this ZIP into its own folder. Copy the entire `data` and `models` folders from your old project into `face_attendance_no_liveness`. That preserves registrations, the trained model, and any attendance history. Do not copy the old Python scripts or config.json into the new folder.

You do not need to register or train again when you copy both folders. Face preprocessing and label mapping are compatible with the original version.

You can also replace ONLY `attendance.py` and add `face_detection.py` in your existing project folder. Its existing config.json and virtual environment work with these files. This lets you retain your current setup; register.py in that folder may still import FaceTracker from liveness.py, but registration never performs challenges and the new attendance.py does not import liveness.py. The clean project supplied here contains no liveness.py.

## Run on macOS

Open Terminal inside the new project folder. Python 3.11 must be installed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_models.py
python attendance.py
```

If using the two-file replacement in your existing folder, just run:

```bash
source .venv/bin/activate
python attendance.py
```

Look toward the camera until your face is recognized consistently. Q quits; R resets recognition. Attendance is saved in `data/attendance.csv` once per student per day using Asia/Dubai time. The file is created after the first successful recognition.

```bash
open data/attendance.csv
```

If the camera is unavailable, grant Terminal camera permission or try `python attendance.py --camera 1`. Run only one instance; the CSV logger does not support concurrent writers. This camera path supports OpenCV-accessible laptop or USB cameras; CSI Raspberry Pi camera integration is not included.

## Start from scratch

If you did not copy existing registrations, run these before attendance:

```bash
python register.py --id S001 --name "Sultan"
python train.py
python attendance.py
```

Registration captures 40 samples. Register other students before training, or rerun training after adding them. Enrollment is supervised and stores unencrypted face images locally.

## Files and algorithm

- register.py captures grayscale, resized, equalized face images.
- train.py trains LBPH and saves models/lbph_model.yml and models/labels.json.
- face_detection.py uses MediaPipe only to locate faces and landmarks for cropping. A landmark detector does not itself check liveness.
- attendance.py compares crops using LBPH, requires consistent identity, and logs attendance.
- common.py shares preprocessing and camera functions.
- config.json contains the three remaining settings.
- setup_models.py downloads the official Face Landmarker model once.

The initial recognition_threshold is 65, an unvalidated starting point. Lower LBPH distance indicates a closer match, not an accuracy percentage. Tune using fresh images of enrolled and unknown people. Lowering the threshold rejects more faces; raising it can increase false acceptance.

## Validation

Five automated tests passed, and the face detector loaded successfully and processed a blank frame on CPU.

Automated tests cover consecutive recognition, unknown/changed identities, daily duplicate prevention, and LBPH model save/load. The MediaPipe model is also checked on a blank frame. Live camera accuracy is not measured here.

```bash
python -m unittest discover -s tests -v
```

Windows: create a virtual environment with `py -3.11 -m venv .venv` and use `.venv\Scripts\python.exe` instead of `python` in the commands above. Do not mix different OpenCV wheels in one environment; use requirements.txt in a clean environment.

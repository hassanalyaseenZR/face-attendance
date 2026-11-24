# Face Attendance Without Liveness (Raspberry Pi)

Face-recognition attendance for Raspberry Pi using LBPH recognition and MediaPipe face cropping. Attendance is recorded after eight consecutive accepted matches for one student. Unknown faces, multiple detected faces, and loss of a face reset the recognition count.

**Warning:** there are no blink, head-turn, or eye checks. This version can accept a photo or video of an enrolled student, so it provides no anti-spoofing. Recognition thresholds still require calibration.

## Requirements

- Raspberry Pi 4 or 5 running **64-bit** Raspberry Pi OS (Bookworm). MediaPipe does not provide wheels for 32-bit OS.
- Python 3.11 (the default on Bookworm). Check with `python3 --version`.
- A USB webcam. CSI ribbon cameras are not supported by this project.
- A monitor connected to the Pi, or VNC into the desktop. The preview window needs a graphical desktop and will not open over plain SSH.

## Install system packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libgl1 libglib2.0-0
sudo usermod -aG video $USER
```

Log out and back in (or reboot) once so the `video` group change takes effect.

Check that the camera is detected:

```bash
ls /dev/video*
```

## Install the project

```bash
git clone https://github.com/hassanalyaseenZR/face-attendance.git
cd face-attendance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_models.py
```

`setup_models.py` downloads the official Face Landmarker model once, so the Pi needs internet for that step. Do not mix different OpenCV packages in one environment. LBPH needs `opencv-contrib-python`, so use the versions in `requirements.txt` in a clean venv.

## Keep existing registrations

If you registered students on another computer, copy the entire `data` and `models` folders into the project folder on the Pi (USB drive or `scp`):

```bash
scp -r data models pi@<pi-ip-address>:~/face-attendance/
```

You do not need to register or train again. Do not copy old Python scripts, `config.json`, or the old `.venv`. Virtual environments are not portable between Mac and Pi.

## Run

```bash
cd face-attendance
source .venv/bin/activate
python attendance.py
```

Look toward the camera until your face is recognized consistently. **Q** quits; **R** resets recognition.

Attendance is saved to `data/attendance.csv`, once per student per day, using Asia/Dubai time regardless of the Pi's system timezone. The file is created after the first successful recognition. View it with:

```bash
cat data/attendance.csv
```

Or open it from the desktop with `xdg-open data/attendance.csv`.

If the camera is not found, try `python attendance.py --camera 1` (or `--camera 2`). Make sure no other program is using the camera. Run only one instance at a time; the CSV logger does not support concurrent writers.

## Start from scratch

If you have no existing registrations:

```bash
python register.py --id S001 --name "Sultan"
python train.py
python attendance.py
```

Registration captures 40 samples. Register all students before training, or rerun `train.py` after adding more. Enrollment is supervised, and face images are stored unencrypted on the Pi's SD card.

## Files and algorithm

- `register.py` captures grayscale, resized, equalized face images.
- `train.py` trains LBPH and saves `models/lbph_model.yml` and `models/labels.json`.
- `face_detection.py` uses MediaPipe only to locate faces and landmarks for cropping. It does not check liveness.
- `attendance.py` compares crops using LBPH, requires consistent identity, and logs attendance.
- `common.py` shares preprocessing and camera functions.
- `config.json` contains the three remaining settings.
- `setup_models.py` downloads the Face Landmarker model.

The initial `recognition_threshold` is 65, an unvalidated starting point. Lower LBPH distance means a closer match, not an accuracy percentage. Tune it using fresh images of enrolled and unknown people. Lowering the threshold rejects more faces; raising it can increase false acceptance.

## Performance tips

- Use good, even lighting facing the person. LBPH is sensitive to lighting changes.
- Register students in the same place and lighting where attendance will run.
- If the preview is slow, lower the camera resolution in `common.py` or use a Pi 5.

## Running automatically at boot (optional)

Because the preview needs the desktop, use the desktop autostart rather than a system service. Create `~/.config/autostart/attendance.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Face Attendance
Exec=/bin/bash -c "cd /home/pi/face-attendance && source .venv/bin/activate && python attendance.py"
```

Change `/home/pi` if your username is different.

## Validation

Automated tests cover consecutive recognition, unknown/changed identities, daily duplicate prevention, and LBPH model save/load. The MediaPipe model is also checked on a blank frame. Live camera accuracy on the Pi has not been measured; test it with your own camera before relying on it.

```bash
python -m unittest discover -s tests -v
```

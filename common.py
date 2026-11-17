"""Small shared functions for camera input, face crops, and configuration."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_config():
    return json.loads((ROOT / 'config.json').read_text())


def open_camera(index):
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Camera unavailable. Check permissions or try --camera 1.')
    return camera


def face_box(landmarks, frame):
    height, width = frame.shape[:2]
    xs = [p.x * width for p in landmarks]
    ys = [p.y * height for p in landmarks]
    x1, x2 = max(0, int(min(xs))), min(width, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(height, int(max(ys)))
    return x1, y1, x2, y2


def face_crop(frame, landmarks):
    x1, y1, x2, y2 = face_box(landmarks, frame)
    if min(x2-x1, y2-y1) < 80:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # Exactly the same preprocessing is used during registration and prediction.
    return cv2.equalizeHist(cv2.resize(gray, (160, 160)))


def show_message(frame, text, line=0, color=(0, 255, 255)):
    y = 28 + line * 28
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def make_recognizer():
    if not hasattr(cv2, 'face'):
        raise RuntimeError('Install opencv-contrib-python; LBPH needs cv2.face.')
    return cv2.face.LBPHFaceRecognizer_create()

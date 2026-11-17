"""Face landmarks for detection and consistent crops only; no liveness checks."""
import time
import cv2
from common import ROOT


class FaceTracker:
    """Read landmarks using the current MediaPipe Tasks API."""
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model = ROOT / 'models' / 'face_landmarker.task'
        if not model.exists():
            raise RuntimeError('Missing landmark model. Run python setup_models.py first.')
        self.mp = mp
        self.last_timestamp = -1
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=2,  # Detect a second face so we can reject the frame.
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6))

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = max(self.last_timestamp + 1, int(time.monotonic() * 1000))
        self.last_timestamp = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(image, timestamp).face_landmarks

    def close(self):
        self.detector.close()


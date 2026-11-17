"""Recognize one student and append one attendance row per day; no liveness."""
import argparse
import csv
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
from common import ROOT, face_box, face_crop, load_config, make_recognizer, open_camera, show_message
from face_detection import FaceTracker


def record_attendance(student, path, now=None):
    """Single-process CSV logger. Return False if already recorded on this local day."""
    now = now or datetime.now(ZoneInfo('Asia/Dubai'))
    today = now.date().isoformat()
    if path.exists():
        with path.open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                if row['student_id'] == student['id'] and row['date'] == today:
                    return False
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['student_id', 'name', 'date', 'time', 'timezone', 'status'])
        writer.writerow([student['id'], student['name'], today,
                         now.strftime('%H:%M:%S'), str(now.tzinfo), 'Present'])
    return True


class VerificationSession:
    """Require consecutive recognition matches for the same student."""
    def __init__(self, config):
        self.required_frames = config['stable_identity_frames']
        self.reset()

    def reset(self):
        self.label = None
        self.frames = 0

    def update(self, label):
        if label is None:
            self.reset()
            return False
        if label != self.label:
            self.reset()
            self.label = label
        self.frames += 1
        return self.frames >= self.required_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    config = load_config()
    model_path = ROOT / 'models' / 'lbph_model.yml'
    labels_path = ROOT / 'models' / 'labels.json'
    if not model_path.exists() or not labels_path.exists():
        raise RuntimeError('No trained model. Register students and run python train.py first.')
    labels = json.loads(labels_path.read_text())
    model = make_recognizer()
    model.read(str(model_path))
    camera = open_camera(args.camera)
    tracker = None
    session = VerificationSession(config)
    notice, notice_until = '', 0
    last_frame_time = None
    try:
        tracker = FaceTracker()
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Camera frame unavailable. Attendance stopped.')
            faces = tracker.detect(frame)
            now = time.monotonic()
            # A camera/processing stall invalidates continuity of recognition.
            if last_frame_time is not None and now-last_frame_time > 1.0:
                session.reset()
            last_frame_time = now
            message = 'Show one registered face'
            debug = ''
            if now < notice_until:
                session.reset()
                message = notice
            elif len(faces) != 1:
                session.reset()
                message = 'Multiple faces: retry alone' if faces else 'No face: verification reset'
            else:
                points = faces[0]
                crop = face_crop(frame, points)
                if crop is None:
                    session.reset()
                    message = 'Move closer to the camera'
                else:
                    label, distance = model.predict(crop)
                    known = str(label) in labels and distance <= config['recognition_threshold']
                    accepted_label = label if known else None
                    passed = session.update(accepted_label)
                    debug = f'LBPH distance: {distance:.1f} (lower is closer)'
                    if not known:
                        message = 'Unknown face: attendance denied'
                    elif passed:
                        student = labels[str(label)]
                        saved = record_attendance(student, ROOT / 'data' / 'attendance.csv')
                        notice = student['name'] + (': attendance recorded' if saved else ': already recorded today')
                        print(notice)
                        notice_until = now + 3
                        message = notice
                        session.reset()
                    else:
                        message = 'Recognizing: ' + labels[str(label)]['name']
                x1, y1, x2, y2 = face_box(points, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            show_message(frame, message)
            show_message(frame, debug, 1)
            show_message(frame, 'Q to quit | R to reset recognition', 2)
            cv2.imshow('Face attendance (unmirrored preview)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                session.reset()
    finally:
        camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))

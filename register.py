"""Capture face samples for one student. Run train.py after registering students."""
import argparse
import json
import re
import time
import cv2
from common import ROOT, face_crop, load_config, open_camera, show_message
from face_detection import FaceTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True, help='Unique student ID, e.g. U22200419')
    parser.add_argument('--name', required=True)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.id) or not args.name.strip():
        parser.error('Use a valid ID (letters, numbers, underscore, hyphen) and nonempty name.')
    students_file = ROOT / 'data' / 'students.json'
    students = json.loads(students_file.read_text()) if students_file.exists() else {}
    folder = ROOT / 'data' / 'faces' / args.id
    if args.id in students or folder.exists():
        parser.error('This ID already exists. Use a new ID; existing samples are not overwritten.')
    config = load_config()
    camera = open_camera(args.camera)
    tracker = None
    samples = []  # Keep samples in memory until registration finishes successfully.
    last_capture = 0
    try:
        tracker = FaceTracker()
        while len(samples) < config['enrollment_images']:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Could not read a camera frame.')
            faces = tracker.detect(frame)
            crop = face_crop(frame, faces[0]) if len(faces) == 1 else None
            if crop is not None and time.monotonic()-last_capture >= 0.25:
                samples.append(crop)
                last_capture = time.monotonic()
            show_message(frame, f'Captured {len(samples)}/{config["enrollment_images"]}')
            show_message(frame, 'Look forward; vary expression and head angle slightly', 1)
            show_message(frame, 'One face only. Q cancels registration.', 2)
            cv2.imshow('Register student (unmirrored preview)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Registration cancelled; no samples saved.')
                return
        folder.mkdir(parents=True)
        for i, sample in enumerate(samples):
            if not cv2.imwrite(str(folder / f'{i:03d}.png'), sample):
                raise RuntimeError('Could not save a face sample.')
        students[args.id] = args.name.strip()
        students_file.write_text(json.dumps(students, indent=2, ensure_ascii=False))
        print(f'Registered {args.name}. Run python train.py next.')
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

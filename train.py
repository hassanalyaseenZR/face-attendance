"""Train an LBPH model on the registered grayscale face samples."""
import json
import cv2
import numpy as np
from common import ROOT, make_recognizer


def main():
    students_file = ROOT / 'data' / 'students.json'
    if not students_file.exists():
        raise RuntimeError('Register students first: python register.py --id ... --name ...')
    students = json.loads(students_file.read_text())
    images, labels, mapping = [], [], {}
    for label, student_id in enumerate(sorted(students)):
        paths = sorted((ROOT / 'data' / 'faces' / student_id).glob('*.png'))
        if len(paths) < 10:
            raise RuntimeError(f'{student_id} needs at least 10 saved face samples.')
        mapping[str(label)] = {'id': student_id, 'name': students[student_id]}
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (160, 160):
                raise RuntimeError(f'Invalid sample: {path.name}')
            images.append(image)
            labels.append(label)
    if not images:
        raise RuntimeError('No training images found.')
    model = make_recognizer()
    model.train(images, np.array(labels, dtype=np.int32))
    (ROOT / 'models').mkdir(exist_ok=True)
    model.write(str(ROOT / 'models' / 'lbph_model.yml'))
    (ROOT / 'models' / 'labels.json').write_text(json.dumps(mapping, indent=2))
    print(f'Trained {len(mapping)} students using {len(images)} samples.')
    print('Next: python attendance.py')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))

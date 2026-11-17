"""Download Google's official MediaPipe Face Landmarker model once."""
from pathlib import Path
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def main():
    destination = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'
    if destination.exists():
        print('Landmark model already exists.')
        return
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        print('Downloading official Face Landmarker model (internet required)...')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as file:
            while chunk := response.read(1024*1024):
                file.write(chunk)
        if temporary.stat().st_size < 1000000:
            raise RuntimeError('Download looks incomplete; retry.')
        temporary.replace(destination)
        print('Model ready.')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

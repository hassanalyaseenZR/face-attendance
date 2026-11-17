import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import load_config, make_recognizer
import numpy as np
from attendance import VerificationSession, record_attendance
from datetime import datetime
from zoneinfo import ZoneInfo

class RecognitionTests(unittest.TestCase):
    def test_stable_matches(self):
        session=VerificationSession(load_config())
        for _ in range(7):
            self.assertFalse(session.update(0))
        self.assertTrue(session.update(0))

    def test_unknown_resets(self):
        session=VerificationSession(load_config())
        for _ in range(7): session.update(0)
        self.assertFalse(session.update(None))
        self.assertFalse(session.update(0))
        self.assertEqual(session.frames,1)

    def test_changed_identity_resets(self):
        session=VerificationSession(load_config())
        for _ in range(7): session.update(0)
        self.assertFalse(session.update(1))
        self.assertEqual(session.frames,1)
        self.assertEqual(session.label,1)

    def test_daily_logging(self):
        student={'id':'S001','name':'Sultan'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data'/'attendance.csv'
            today=datetime(2026,10,5,19,tzinfo=ZoneInfo('Asia/Dubai'))
            tomorrow=datetime(2026,10,6,19,tzinfo=ZoneInfo('Asia/Dubai'))
            self.assertTrue(record_attendance(student,path,today))
            self.assertFalse(record_attendance(student,path,today))
            self.assertTrue(record_attendance(student,path,tomorrow))
            self.assertEqual(len(path.read_text().splitlines()),3)

    def test_model_save_load(self):
        rng=np.random.default_rng(42)
        samples=[rng.integers(0,256,(160,160),dtype=np.uint8) for _ in range(2)]
        model=make_recognizer()
        model.train(samples,np.array([0,1],dtype=np.int32))
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'model.yml')
            model.write(path)
            restored=make_recognizer()
            restored.read(path)
            for expected,sample in enumerate(samples):
                label,distance=restored.predict(sample)
                self.assertEqual(label,expected)
                self.assertAlmostEqual(distance,0)

if __name__=='__main__': unittest.main()

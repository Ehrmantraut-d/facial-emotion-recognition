"""MediaPipe face detection with a local OpenCV Haar fallback."""
from __future__ import annotations

import logging
import hashlib
import shutil
import urllib.request
from pathlib import Path

import cv2

from src.config import MODEL_DIR

LOG = logging.getLogger(__name__)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_PATH = MODEL_DIR / "blaze_face_short_range.tflite"
MODEL_SHA256 = "B4578F35940BF5A1A655214A1CCE5CAB13EBA73C1297CD78E1A04C2380B0152F"


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _ensure_model() -> Path:
    """Fetch Google's small face-detection model once; never sends camera frames."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000 and _sha256(MODEL_PATH)==MODEL_SHA256:
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".tflite.part")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "facial-emotion-ai/1.0"})
        with urllib.request.urlopen(request, timeout=25) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 100_000 or _sha256(temporary)!=MODEL_SHA256:
            raise RuntimeError("Downloaded MediaPipe model failed size or SHA-256 validation")
        temporary.replace(MODEL_PATH)
        return MODEL_PATH
    finally:
        temporary.unlink(missing_ok=True)


class FaceDetector:
    def __init__(self):
        self.backend = "haar"
        self._mp_legacy = None
        self._mp_task = None
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                self._mp_legacy = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.55)
                self.backend = "mediapipe-solutions"
            else:
                asset = _ensure_model()
                options = mp.tasks.vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    min_detection_confidence=0.55)
                self._mp_task = mp.tasks.vision.FaceDetector.create_from_options(options)
                self.backend = "mediapipe-tasks"
        except Exception as exc:
            LOG.warning("MediaPipe face detector unavailable; using Haar fallback: %s", exc)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError("Neither MediaPipe nor OpenCV Haar face detection is available") from exc

    @staticmethod
    def _clip_box(x: int, y: int, width: int, height: int, frame_w: int, frame_h: int):
        x, y = max(0, x), max(0, y)
        width, height = min(frame_w - x, width), min(frame_h - y, height)
        return (x, y, width, height) if width > 0 and height > 0 else None

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self._mp_legacy is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_legacy.process(rgb)
            boxes = []
            for item in (res.detections or []):
                b = item.location_data.relative_bounding_box
                box = self._clip_box(int(b.xmin*w), int(b.ymin*h), int(b.width*w), int(b.height*h), w, h)
                if box: boxes.append(box)
            return boxes
        if self._mp_task is not None:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mp_task.detect(image)
            boxes=[]
            for item in result.detections:
                b=item.bounding_box
                box=self._clip_box(b.origin_x,b.origin_y,b.width,b.height,w,h)
                if box: boxes.append(box)
            return boxes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [tuple(map(int, box)) for box in self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))]

    def close(self):
        if self._mp_legacy is not None: self._mp_legacy.close()
        if self._mp_task is not None: self._mp_task.close()

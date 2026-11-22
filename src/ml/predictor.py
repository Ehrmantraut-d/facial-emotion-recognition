"""Persisted-model inference helpers shared by the GUI and evaluation scripts."""
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.ml.models import predict_with_flip_average


class EmotionPredictor:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}. Add the supplied CNN checkpoint or prepare FER2013 data and run scripts/train_cnn.py."
            )
        bundle = joblib.load(path)
        self.model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        self.inference_strategy = bundle.get("inference_strategy", "single_view") if isinstance(bundle, dict) else "single_view"
        self.flip_average = self.inference_strategy == "horizontal_flip_probability_average"
        self.class_ids = np.asarray(getattr(self.model, "classes_", np.arange(7)))

    def predict_face(self, face: np.ndarray) -> tuple[int, float, dict[int, float]]:
        """Convert a face crop to training-compatible pixels and return class scores."""
        if face is None or face.size == 0:
            raise ValueError("Face crop is empty")
        gray = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        features = crop.reshape(1, -1)
        predictions, probabilities = predict_with_flip_average(
            self.model, features, enabled=self.flip_average
        )
        prediction = int(predictions[0])
        scores = {int(class_id): float(score) for class_id, score in zip(self.class_ids, probabilities[0])}
        return prediction, scores.get(prediction, 0.0), scores

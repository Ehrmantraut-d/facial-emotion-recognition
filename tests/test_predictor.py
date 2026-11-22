import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from src.ml.predictor import EmotionPredictor

def test_predictor_loads_and_returns_probabilities(tmp_path):
    X=np.random.default_rng(0).random((14,2304)); y=np.arange(14)%7
    model=KNeighborsClassifier(n_neighbors=1).fit(X,y)
    path=tmp_path/"m.joblib"; joblib.dump({"model":model},path)
    predictor=EmotionPredictor(path)
    label,score,scores=predictor.predict_face(np.zeros((48,48),dtype=np.uint8))
    assert label in range(7) and 0 <= score <= 1 and set(scores)==set(range(7))

def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError): EmotionPredictor(tmp_path/"missing.joblib")

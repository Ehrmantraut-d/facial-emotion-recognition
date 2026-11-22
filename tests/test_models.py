import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.ml.models import fit_and_score

def test_training_metrics_and_class_count():
    rng=np.random.default_rng(2)
    X=rng.random((70,12)); y=np.arange(70)%7
    result=fit_and_score("knn",KNeighborsClassifier(n_neighbors=3),X[:56],y[:56],X[56:],y[56:])
    assert 0 <= result["accuracy"] <= 1
    assert len(result["confusion_matrix"]) == 7
    assert result["train_samples"] == 56

def test_small_mlp_trains_and_returns_seven_class_probabilities():
    rng=np.random.default_rng(4)
    X=rng.random((70,12)); y=np.arange(70)%7
    model=MLPClassifier(hidden_layer_sizes=(8,),max_iter=3,random_state=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result=fit_and_score("mlp-test",model,X[:56],y[:56],X[56:],y[56:])
    assert result["train_samples"] == 56
    assert result["confusion_matrix"] and result["accuracy"] >= 0
    assert model.predict_proba(X[56:57]).shape == (1,7)

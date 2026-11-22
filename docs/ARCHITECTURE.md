# Architecture

```text
FER2013 CSV -> dataset validation -> normalized arrays -> EDA and splits
                                         |-> KNN / MLP / FER+ CNN -> validation selection
                                                                -> final test report -> joblib bundle
webcam -> MediaPipe (or Haar fallback) -> 48x48 grayscale -> selected CNN predict_proba -> Tkinter
```

`src/data` owns acquisition and validation; `src/ml` contains scikit-learn estimators and inference; `src/vision` detects faces; `src/gui` renders the desktop experience; `src/tracking.py` writes per-run evidence. `scripts/run_pipeline.py` orchestrates the phases. Paths derive from the repository root in `src/config.py`, not the current working directory. The webcam stream stays local.

Training features are grayscale pixels scaled to [0,1] and flattened in row-major order. Webcam crops use grayscale conversion, resize to 48x48, and the same scale and flattening. The detector only localizes faces; it does not classify expression.

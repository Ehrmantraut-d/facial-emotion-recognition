# Real-Time Facial Expression Recognition

Academic project for training and comparing KNN, MLP, and convolutional neural networks on FER2013/FER+, then using the validation-selected classifier in a local Tkinter webcam application. It predicts visible facial-expression labels; it does not infer a person's internal emotional state.

## Quick start

1. Clone the repository and install Conda (Anaconda, Miniconda, or Miniforge).
2. From the project folder, run `conda env create -f environment.yml` and `conda activate facial_emotion_ai`.
3. Launch the included CNN with `Open_FER_Studio.bat` on Windows or `sh Open_FER_Studio.sh` on macOS/Linux. You can also run `python main.py` from the activated environment.

The selected CNN checkpoint is included, so first-time inference does not require downloading FER2013. For a local venv without Conda, use Python 3.11 and run `python scripts/setup_environment.py` on Windows or `python3 scripts/setup_environment.py` on macOS/Linux. CPU PyTorch is installed by default for portability; CUDA is optional.

## Retraining

1. Obtain the original `fer2013.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data), which requires a Kaggle account and acceptance of its rules. Configure the official Kaggle CLI or place the CSV in `data/raw/fer2013/fer2013.csv`.
2. Run `python scripts/prepare_dataset.py` to validate the official splits and create normalized arrays locally.
3. Run `python scripts/train_ferplus.py` to download Microsoft’s aligned FER+ vote file and train the label-consensus MLP, then `python scripts/train_cnn.py` to train and select the CNN.

The project does not commit the FER2013 face images, processed arrays, or run-specific training intermediates. `.gitignore` keeps those machine-local. A FER+ CNN checkpoint and compact evaluation artifacts are included for immediate app use and reproducibility.

## Workflow and outputs

The pipeline downloads or reuses the original CSV, validates labels and 48×48 pixels, and respects the official `Training`, `PublicTest`, and `PrivateTest` partitions. Dense baselines use flattened, normalized pixels; the CNN keeps the image axes and normalized grayscale pixels. Metrics and run artifacts are stored under `results/`, processed arrays under `data/processed/`, and selected model metadata under `models/`. Existing runs are preserved.

`python scripts/run_pipeline.py --mode quick` uses a reproducible stratified subset and marks all results as development-only. `--mode full` compares KNN candidates and an MLP. The separate FER+ experiment trains a spatial CNN on aligned 48×48 face images; its validation-selected weights are used in the GUI and its held-out test partition is evaluated once.

Open `notebooks/01_dataset_and_results.ipynb`, `notebooks/02_ml_results_and_ferplus.ipynb`, or `notebooks/03_cnn_layers_results.ipynb` in Jupyter. They contain saved result tables and the CNN training history, and read project artifacts without downloading data implicitly.

## Development and diagnostics

- `python scripts/check_environment.py` writes `results/environment/environment_check.json`.
- `pytest -q` runs offline tests; webcam access is mocked.
- `python main.py` or double-click `Open_FER_Studio.bat` to run the Tkinter application. Face detection uses MediaPipe Tasks and Google's small short-range face detector model; first launch downloads this model from Google's official storage. OpenCV Haar detection is the offline fallback. Webcam frames stay on the device and are not saved or transmitted.
- The source CSV is reused when already present; every invocation creates a new result run directory.

## Current execution status

The active CNN achieved **0.7378 accuracy / 0.5159 macro-F1** on 3,356 accepted PublicTest consensus rows, up from 0.5498 / 0.3636 for the FER+ MLP on those same rows. On 3,329 accepted PrivateTest consensus rows, it scored **0.7188 accuracy / 0.5081 macro-F1**. These scores use filtered FER+ consensus targets and are not comparable to original-label FER2013 scores. See `results/cnn_ferplus/` and `docs/PROJECT_REPORT.md` for details.

FER2013 images depict faces and the dataset is imbalanced and noisy. Check the original provider's terms before use or redistribution. Evaluation on FER2013 does not establish generalization to live webcam lighting, populations, or real-world contexts. The test set is reserved for the final selected model only.

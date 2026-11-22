# User guide

## Install

From this folder, create the Conda environment with `conda env create -f environment.yml`, then `conda activate facial_emotion_ai`. The same environment file supports Windows, Linux, and macOS. In Anaconda Navigator, choose the `facial_emotion_ai` environment to launch Jupyter Notebook. Without Conda, use Python 3.11 and `python scripts/setup_environment.py` (or `python3` on macOS/Linux) to create an isolated `.venv` and register a kernel.

## Acquire and train

Configure the official Kaggle CLI and accept the competition rules, or place the authorized original `fer2013.csv` in `data/raw/fer2013/`. Run `python scripts/prepare_dataset.py` to generate the full normalized split arrays. Then run `python scripts/train_ferplus.py` and `python scripts/train_cnn.py`. The original-label KNN/MLP comparison is available through `python scripts/run_pipeline.py --mode full`; quick mode is development-only.

## Inspect results

Runs are stored in unique timestamped folders under `results/runs/`. A compact cross-computer copy of the active CNN evaluation and history is saved under `results/cnn_ferplus/`. `models/selected_model.json` records the active CNN and test metrics. Open `notebooks/03_cnn_layers_results.ipynb` for the saved evaluation.

## Launch the webcam app

Run `python main.py`, or use the OS-specific `Open_FER_Studio.bat` / `Open_FER_Studio.sh` launcher. Press **Démarrer la caméra** and **Arrêter** to control capture. Close the window to release the camera. If the default camera is unavailable, update `camera_index` when constructing `EmotionApp`. MediaPipe Tasks uses Google's short-range face detector asset (downloaded on first launch); OpenCV Haar is the offline fallback. Webcam frames are not recorded or transmitted.

## Scientific interpretation

The score is the classifier's output probability, not confidence that the person feels a particular emotion. It is a visible-expression estimate with known dataset and domain limitations.

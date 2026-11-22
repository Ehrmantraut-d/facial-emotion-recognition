# FER2013 data

Place the original `fer2013.csv` here: `data/raw/fer2013/fer2013.csv`. The expected CSV has columns `emotion`, `pixels`, and `Usage`, with the official `Training`, `PublicTest`, and `PrivateTest` values. `scripts/run_pipeline.py` downloads from the [original Kaggle competition data page](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data) through its authenticated CLI when configured. Access requires a Kaggle account and acceptance of competition rules. The project does not redistribute face images or assume an unrestricted dataset license.

To import a ZIP you already downloaded without unpacking unrelated files, run `python scripts/import_dataset_archive.py --archive "/path/to/dataset.zip"` (macOS/Linux) or `python scripts/import_dataset_archive.py --archive "C:\\path\\to\\dataset.zip"` (Windows).

Generated normalized arrays and validation metadata are local outputs and are excluded from version control.

FER+ vote labels are downloaded from the official [Microsoft FERPlus repository](https://github.com/microsoft/FERPlus) automatically when `python scripts/train_ferplus.py` runs and the local CSV is absent. They align row-for-row with FER2013 and add crowd-sourced votes rather than new images. Training retains rows with a unique highest vote among the seven target expressions and at least four votes. Non-target labels are not outputs in this project’s seven-class model. Retain the original Kaggle terms for the image data; Microsoft does not host the original images.

"""FER2013 acquisition, validation, preprocessing, and persistence."""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (CLASS_NAMES, IMAGE_SHAPE, METADATA_DIR, PROCESSED_DIR,
                        RAW_DIR, RANDOM_SEED)

LOG = logging.getLogger(__name__)
KAGGLE_COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
SOURCE_URL = f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/data"


def acquire_dataset() -> Path:
    """Return CSV path, downloading through authenticated Kaggle CLI if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(RAW_DIR.glob("*.csv"))
    if existing:
        return existing[0]
    cli = shutil.which("kaggle")
    if not cli:
        candidate = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
        if candidate.is_file(): cli = str(candidate)
    if not cli:
        raise FileNotFoundError(
            f"FER2013 CSV not found in {RAW_DIR}. Install/configure the official Kaggle CLI "
            f"or download the dataset from {SOURCE_URL} and place fer2013.csv there."
        )
    result = subprocess.run([cli, "competitions", "download", "-c", KAGGLE_COMPETITION,
                             "-p", str(RAW_DIR)], capture_output=True, text=True,
                            timeout=900, check=False)
    if result.returncode:
        details=(result.stderr+"\n"+result.stdout).strip()
        raise RuntimeError(f"Kaggle download failed ({result.returncode}): {details[-1800:] or 'No CLI details returned.'}")
    for archive in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            root = RAW_DIR.resolve()
            for info in zf.infolist():
                dest = (RAW_DIR / info.filename).resolve()
                if not dest.is_relative_to(root):
                    raise RuntimeError("ZIP archive contains an unsafe path")
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, dest.open("wb") as target:
                        shutil.copyfileobj(source, target)
    for archive in RAW_DIR.glob("*.tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            root = RAW_DIR.resolve()
            for member in tf.getmembers():
                dest = (RAW_DIR / member.name).resolve()
                if not dest.is_relative_to(root):
                    raise RuntimeError("TAR archive contains an unsafe path")
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    source = tf.extractfile(member)
                    if source is None: continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with source, dest.open("wb") as target:
                        shutil.copyfileobj(source, target)
    csvs = list(RAW_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError("Kaggle download completed but contained no CSV file.")
    return csvs[0]


def _parse_pixels(value: object) -> np.ndarray:
    """Parse one FER2013 space-separated pixel string into a validated uint8 image."""
    arr = np.fromstring(str(value), sep=" ", dtype=np.float32)
    if arr.size != IMAGE_SHAPE[0] * IMAGE_SHAPE[1] or not np.isfinite(arr).all():
        raise ValueError("pixel vector must contain 2304 finite values")
    if (arr < 0).any() or (arr > 255).any():
        raise ValueError("pixel intensity must be in [0,255]")
    return arr.astype(np.uint8).reshape(IMAGE_SHAPE)


def load_and_validate(csv_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load official splits and report malformed records without losing evidence."""
    records = {"Training": [], "PublicTest": [], "PrivateTest": []}
    invalid = []
    total = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels", "Usage"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required)}; got {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            total += 1
            try:
                label = int(row["emotion"])
                if label not in range(len(CLASS_NAMES)):
                    raise ValueError(f"label {label} outside 0..6")
                split = row["Usage"].strip()
                if split not in records:
                    raise ValueError(f"unknown split {split!r}")
                records[split].append((_parse_pixels(row["pixels"]), label))
            except (ValueError, TypeError) as exc:
                invalid.append({"line": line_no, "reason": str(exc)})
    if not records["Training"] or not records["PublicTest"] or not records["PrivateTest"]:
        raise ValueError("CSV must contain nonempty Training, PublicTest, and PrivateTest partitions")
    arrays = {name: (np.stack([r[0] for r in rows]), np.asarray([r[1] for r in rows], dtype=np.int64))
              for name, rows in records.items()}
    report = {"source_file": csv_path.name, "source_url": SOURCE_URL, "rows_seen": total,
              "invalid_count": len(invalid), "invalid_examples": invalid[:50],
              "dimensions": [48, 48], "class_names": list(CLASS_NAMES),
              "split_counts": {k: len(v[1]) for k, v in arrays.items()},
              "label_counts": {split: {CLASS_NAMES[i]: int((y == i).sum()) for i in range(7)}
                               for split, (_, y) in arrays.items()}}
    return arrays, report


def prepare_data(csv_path: Path | None = None, quick: bool = False,
                 run_dir: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Prepare normalized flat features; fit split only from official training data."""
    csv_path = csv_path or acquire_dataset()
    splits, report = load_and_validate(csv_path)
    Xtr, ytr = splits["Training"]
    Xtest, ytest = splits["PrivateTest"]
    Xval, yval = splits["PublicTest"]
    # Preserve the competition's public validation and private test partitions.
    if quick:
        n = min(700, len(ytr))
        idx, _ = train_test_split(np.arange(len(ytr)), train_size=n, random_state=RANDOM_SEED,
                                  stratify=ytr)
        Xtr, ytr = Xtr[idx], ytr[idx]
        nv = min(350, len(yval)); nt = min(350, len(ytest))
        iv, _ = train_test_split(np.arange(len(yval)), train_size=nv, random_state=RANDOM_SEED, stratify=yval)
        it, _ = train_test_split(np.arange(len(ytest)), train_size=nt, random_state=RANDOM_SEED, stratify=ytest)
        Xval, yval = Xval[iv], yval[iv]; Xtest, ytest = Xtest[it], ytest[it]
    processed_dir = (run_dir / "processed_data") if quick and run_dir else PROCESSED_DIR
    metadata_dir = (run_dir / "metadata") if quick and run_dir else METADATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True); metadata_dir.mkdir(parents=True, exist_ok=True)
    result = {"train": (Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0, ytr),
              "val": (Xval.reshape(len(Xval), -1).astype(np.float32) / 255.0, yval),
              "test": (Xtest.reshape(len(Xtest), -1).astype(np.float32) / 255.0, ytest)}
    for name, (X, y) in result.items():
        np.save(processed_dir / f"X_{name}.npy", X); np.save(processed_dir / f"y_{name}.npy", y)
    report["prepared_counts"] = {name: len(y) for name, (_, y) in result.items()}
    report["quick_development_subset"] = quick
    report["pixel_range_after_normalization"] = [0.0, 1.0]
    (metadata_dir / "data_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (metadata_dir / "dataset_info.json").write_text(json.dumps({"source": SOURCE_URL,
        "download_source": "Kaggle dataset mirror msambare/fer2013; original FER2013 competition format",
        "license_note": "Check Kaggle competition/provider terms; no blanket redistribution license assumed.",
        "quick_development_subset": quick, "counts": report["prepared_counts"]}, indent=2), encoding="utf-8")
    (metadata_dir / "class_mapping.json").write_text(json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)}, indent=2), encoding="utf-8")
    return result

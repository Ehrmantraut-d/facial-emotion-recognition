"""Train and compare a spatial CNN using aligned FER+ consensus labels."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES, MODEL_DIR, RESULTS_DIR
from src.ml.models import predict_with_flip_average
from scripts.train_ferplus import load_targets


def metrics(model, X, y, flip=True):
    pred, _ = predict_with_flip_average(model, X, enabled=flip)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(7)).tolist(),
            "classification_report": classification_report(
                y, pred, labels=np.arange(7), target_names=CLASS_NAMES,
                zero_division=0, output_dict=True)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-votes", type=int, default=4)
    args = parser.parse_args()
    labels = pd.read_csv(ROOT / "data/raw/ferplus/fer2013new.csv")
    print("Loaded FER+ annotations", flush=True)
    image_rows = pd.read_csv(ROOT / "data/raw/fer2013/fer2013.csv", usecols=["Usage"])
    if len(labels) != len(image_rows) or len(labels) != 35887:
        raise ValueError("FER+ annotations no longer match the official FER2013 row count")
    split_rows, audit = load_targets(ROOT / "data/raw/ferplus/fer2013new.csv", args.min_votes)
    print("Checked FER+ row alignment", flush=True)
    arrays = {key: np.load(ROOT / f"data/processed/X_{key}.npy", mmap_mode="r")
              for key in ("train", "val", "test")}
    mapped = {}
    for split, key in (("Training", "train"), ("PublicTest", "val"), ("PrivateTest", "test")):
        original_ix = np.flatnonzero(image_rows["Usage"].astype(str).to_numpy() == split)
        if len(original_ix) != len(arrays[key]):
            raise ValueError(f"Cached {split} face rows do not align with FER+ labels")
        selected_rows, target, strength = split_rows[split]
        mapped[split] = (np.searchsorted(original_ix, selected_rows), target, strength)
    ix_train, y_train, _ = mapped["Training"]
    ix_val, y_val, val_strength = mapped["PublicTest"]
    X_train, X_val = np.asarray(arrays["train"][ix_train]), np.asarray(arrays["val"][ix_val])
    print(f"Prepared accepted face images: train={len(y_train):,}, validation={len(y_val):,}", flush=True)

    # Compare on identical PublicTest FER+ consensus examples, including the
    # established inference strategy of the currently selected baseline.
    current_meta = json.loads((MODEL_DIR / "selected_model.json").read_text(encoding="utf-8"))
    prior_run_value = current_meta.get("inference_comparison_run")
    if not prior_run_value:
        raise ValueError("Train scripts/train_ferplus.py first so the current model has FER+ consensus validation metrics")
    prior_run = ROOT / prior_run_value
    cnn_validation_path = prior_run / "validation_metrics.json"
    mlp_validation_path = prior_run / "validation_candidate_on_ferplus_consensus.json"
    if cnn_validation_path.is_file():
        baseline_val = json.loads(cnn_validation_path.read_text(encoding="utf-8"))["cnn_candidate"]
    elif mlp_validation_path.is_file():
        baseline_val = json.loads(mlp_validation_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Selected model's FER+ validation metrics were not found in {prior_run}")
    # Reuse the already-computed score on these exact FER+ consensus rows.
    # This also avoids loading both Intel MKL and the PyTorch CPU OpenMP runtime
    # into the same Windows process before the candidate evaluation.
    baseline_flip = current_meta.get("inference_strategy") == "horizontal_flip_probability_average"
    print(f"Baseline measured: accuracy={baseline_val['accuracy']:.4f}; starting CNN", flush=True)
    # Load torch only after sklearn's OpenMP/MKL baseline prediction. On some
    # Windows builds, loading both libraries before the MLP prediction can
    # trigger a runtime DLL initialization failure.
    import torch
    from src.ml.cnn import train_cnn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}; device={device}", flush=True)
    start = time.perf_counter()
    candidate, history = train_cnn(X_train, y_train, X_val, y_val,
                                   epochs=args.epochs, patience=args.patience, device=device)
    elapsed = time.perf_counter() - start
    cnn_val = metrics(candidate, X_val, y_val, flip=True)
    selected = (cnn_val["accuracy"], cnn_val["macro_f1"]) > (
        baseline_val["accuracy"], baseline_val["macro_f1"])

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run = RESULTS_DIR / "runs" / f"{timestamp}_cnn_ferplus"
    suffix = 2
    while run.exists():
        run = RESULTS_DIR / "runs" / f"{timestamp}_cnn_ferplus_{suffix}"
        suffix += 1
    run.mkdir(parents=True)
    audit["counts"]["Training"]["used_for_cnn"] = len(y_train)
    (run / "ferplus_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (run / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    candidate_bundle = {"model": candidate, "name": "ferplus_spatial_cnn_16_32_64",
                        "class_names": list(CLASS_NAMES), "mode": "full_ferplus_cnn",
                        "inference_strategy": "horizontal_flip_probability_average",
                        "validation_accuracy": cnn_val["accuracy"],
                        "validation_macro_f1": cnn_val["macro_f1"],
                        "architecture": "3 convolution blocks (16,32,64), 3x3 conv per block, batch normalization, ReLU, max pooling, adaptive 3x3 pooling, 128-unit dense head; 48x48 grayscale input; train-only random horizontal flips",
                        "run": str(run.relative_to(ROOT)),
                        "inference_comparison_run": str(run.relative_to(ROOT))}
    joblib.dump(candidate_bundle, run / "candidate_model.joblib", compress=3)
    (run / "validation_metrics.json").write_text(json.dumps({
        "baseline_current_model": baseline_val, "cnn_candidate": cnn_val,
        "baseline_flip_average": baseline_flip, "cnn_flip_average": True,
        "validation_rows": len(y_val), "validation_selected_cnn": selected,
        "training_seconds": elapsed, "device": device}, indent=2), encoding="utf-8")

    report = ["# Convolutional facial expression experiment", "",
              "The CNN receives the aligned FER2013 48×48 grayscale face images and learns spatial filters over facial features. Training uses FER+ unique top-vote consensus labels (minimum four votes), with horizontal flips applied only to training batches. PublicTest is used for model selection; PrivateTest remains untouched unless the CNN is selected.", "",
              f"Architecture: {candidate_bundle['architecture']}", "",
              "| Model | PublicTest consensus accuracy | Macro-F1 |", "|---|---:|---:|"]
    report.append(f"| Existing selected model | {baseline_val['accuracy']:.4f} | {baseline_val['macro_f1']:.4f} |")
    report.append(f"| Spatial CNN + flip averaging | {cnn_val['accuracy']:.4f} | {cnn_val['macro_f1']:.4f} |")
    report.extend(["", f"Training stopped after {len(history)} epochs ({elapsed:.1f}s on {device}). CNN selected by validation accuracy, with macro-F1 as tie-breaker: **{selected}**."])
    if selected:
        ix_test, y_test, _ = mapped["PrivateTest"]
        X_test = np.asarray(arrays["test"][ix_test])
        test_metrics = metrics(candidate, X_test, y_test, flip=True)
        (run / "final_test_metrics_ferplus_consensus.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8")
        # Keep confusion counts in JSON. Matplotlib's Windows native backend
        # can conflict with PyTorch's CPU runtime late in this process.
        report.append(f"\nPrivateTest consensus rows evaluated once after selection: {len(y_test):,}; accuracy **{test_metrics['accuracy']:.4f}**, macro-F1 **{test_metrics['macro_f1']:.4f}**.")
        backup = MODEL_DIR / "selected_model_before_cnn.joblib"
        if not backup.exists():
            shutil.copy2(MODEL_DIR / "selected_model.joblib", backup)
        metadata_path = MODEL_DIR / "selected_model.json"
        backup_json = MODEL_DIR / "selected_model_before_cnn.json"
        if metadata_path.exists() and not backup_json.exists():
            shutil.copy2(metadata_path, backup_json)
        stable_path = MODEL_DIR / "selected_model_cnn.joblib"
        joblib.dump(candidate_bundle, stable_path, compress=3)
        shutil.copy2(stable_path, MODEL_DIR / "selected_model.joblib")
        metadata = {k: v for k, v in candidate_bundle.items() if k != "model"}
        metadata.update({"test_metrics_ferplus_consensus": test_metrics,
                         "baseline_validation_ferplus_consensus": baseline_val,
                         "cnn_training_epochs": len(history),
                         "cnn_training_seconds": elapsed})
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        report.append("\nThe CNN did not exceed the current selected model on PublicTest accuracy (macro-F1 breaks accuracy ties). The existing model remains active; the complete CNN weights and metrics are saved in this run for inspection.")
    (run / "model_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if selected:
        stable_dir = RESULTS_DIR / "cnn_ferplus"
        stable_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ferplus_data_audit.json", "validation_metrics.json",
                     "training_history.json", "final_test_metrics_ferplus_consensus.json",
                     "model_comparison_report.md"):
            shutil.copy2(run / name, stable_dir / name)
        metadata["inference_comparison_run"] = str(stable_dir.relative_to(ROOT))
        metadata["run"] = str(stable_dir.relative_to(ROOT))
        metadata["training_run"] = str(run.relative_to(ROOT))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"CNN validation: accuracy={cnn_val['accuracy']:.4f}, macro_f1={cnn_val['macro_f1']:.4f}")
    print(f"Baseline validation: accuracy={baseline_val['accuracy']:.4f}, macro_f1={baseline_val['macro_f1']:.4f}")
    print(f"Selected CNN: {selected}; run: {run}")
    if selected:
        print(f"CNN PrivateTest: accuracy={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

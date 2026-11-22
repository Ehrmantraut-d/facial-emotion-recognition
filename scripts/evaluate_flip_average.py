"""Evaluate validation-selected horizontal-flip averaging without retraining."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import MODEL_DIR, RESULTS_DIR
from src.ml.models import final_metrics
from src.tracking import new_run, plot_confusion, save_json


def main() -> None:
    model_path = MODEL_DIR / "selected_model.joblib"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    X_val = np.load(ROOT / "data" / "processed" / "X_val.npy")
    y_val = np.load(ROOT / "data" / "processed" / "y_val.npy")
    X_test = np.load(ROOT / "data" / "processed" / "X_test.npy")
    y_test = np.load(ROOT / "data" / "processed" / "y_test.npy")

    val_single = final_metrics(model, X_val, y_val)
    val_flip = final_metrics(model, X_val, y_val, flip_average=True)
    # This inference strategy was selected on PublicTest validation. Evaluate the
    # selected strategy once on the untouched PrivateTest partition.
    test_flip = final_metrics(model, X_test, y_test, flip_average=True)

    run = new_run("flip_average")
    save_json(run / "validation_single_view.json", val_single)
    save_json(run / "validation_flip_average.json", val_flip)
    save_json(run / "final_test_metrics.json", test_flip)
    plot_confusion(test_flip["confusion_matrix"], "MLP with horizontal-flip probability average · PrivateTest",
                   run / "final_test_confusion_matrix.png")
    report = (
        "# Horizontal-flip inference comparison\n\n"
        "The selected MLP weights are unchanged. Validation-only comparison selected averaging the class probabilities "
        "from the original 48×48 face and its horizontal mirror. The private test partition was evaluated once with "
        "that selected inference strategy.\n\n"
        f"| Validation method | Accuracy | Macro-F1 |\n|---|---:|---:|\n"
        f"| Single view | {val_single['accuracy']:.4f} | {val_single['macro_f1']:.4f} |\n"
        f"| Horizontal-flip average | {val_flip['accuracy']:.4f} | {val_flip['macro_f1']:.4f} |\n\n"
        f"PrivateTest with horizontal-flip averaging: accuracy **{test_flip['accuracy']:.4f}**, "
        f"macro-F1 **{test_flip['macro_f1']:.4f}** on {len(y_test):,} examples.\n"
    )
    (run / "model_comparison_report.md").write_text(report, encoding="utf-8")

    baseline_model = MODEL_DIR / "selected_model_before_flip_average.joblib"
    baseline_metadata = MODEL_DIR / "selected_model_before_flip_average.json"
    if not baseline_model.exists():
        shutil.copy2(model_path, baseline_model)
    metadata_path = MODEL_DIR / "selected_model.json"
    if metadata_path.exists() and not baseline_metadata.exists():
        shutil.copy2(metadata_path, baseline_metadata)

    bundle["inference_strategy"] = "horizontal_flip_probability_average"
    bundle["validation_accuracy"] = val_flip["accuracy"]
    bundle["validation_macro_f1"] = val_flip["macro_f1"]
    bundle["inference_comparison_run"] = str(run.relative_to(ROOT))
    improved_path = MODEL_DIR / "selected_model_flip_average.joblib"
    joblib.dump(bundle, improved_path, compress=3)
    shutil.copy2(improved_path, model_path)
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata["test_metrics"] = test_flip
    metadata["baseline_test_metrics"] = json.loads((
        RESULTS_DIR / "runs" / "20261121T222130Z_full_6297687c" / "final_test_metrics.json"
    ).read_text(encoding="utf-8"))
    metadata["run"] = str(run.relative_to(ROOT))
    save_json(metadata_path, metadata)
    print(f"Validation single-view accuracy: {val_single['accuracy']:.4f}")
    print(f"Validation flip-average accuracy: {val_flip['accuracy']:.4f}")
    print(f"PrivateTest flip-average accuracy: {test_flip['accuracy']:.4f}")
    print(f"PrivateTest flip-average macro-F1: {test_flip['macro_f1']:.4f}")
    print(f"Run artifacts: {run}")


if __name__ == "__main__":
    main()

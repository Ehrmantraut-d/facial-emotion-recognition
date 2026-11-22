"""Promote a CNN candidate from saved experiment artifacts without retraining."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing candidate_model.joblib")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    val = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((run / "final_test_metrics_ferplus_consensus.json").read_text(encoding="utf-8"))
    history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
    if not val["validation_selected_cnn"]:
        raise SystemExit("The validation report did not select this CNN candidate.")

    current_model = MODEL_DIR / "selected_model.joblib"
    current_metadata = MODEL_DIR / "selected_model.json"
    for source, backup in ((current_model, MODEL_DIR / "selected_model_before_cnn.joblib"),
                           (current_metadata, MODEL_DIR / "selected_model_before_cnn.json")):
        if source.exists() and not backup.exists():
            shutil.copy2(source, backup)

    stable_model = MODEL_DIR / "selected_model_cnn.joblib"
    shutil.copy2(run / "candidate_model.joblib", stable_model)
    shutil.copy2(stable_model, current_model)
    metadata = {
        "name": "ferplus_spatial_cnn_16_32_64",
        "class_names": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "mode": "full_ferplus_cnn",
        "inference_strategy": "horizontal_flip_probability_average",
        "architecture": "3 convolution blocks (16,32,64), 3x3 conv per block, batch normalization, ReLU, max pooling, adaptive 3x3 pooling, 128-unit dense head; 48x48 grayscale input; train-only random horizontal flips",
        "validation_accuracy": val["cnn_candidate"]["accuracy"],
        "validation_macro_f1": val["cnn_candidate"]["macro_f1"],
        "inference_comparison_run": "results/cnn_ferplus",
        "run": "results/cnn_ferplus",
        "training_run": str(run.relative_to(ROOT)),
        "test_metrics_ferplus_consensus": test,
        "baseline_validation_ferplus_consensus": val["baseline_current_model"],
        "cnn_training_epochs": len(history),
        "cnn_training_seconds": val["training_seconds"],
        "training_device": val["device"],
    }
    current_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = [
        "# Convolutional facial expression experiment", "",
        "The CNN receives aligned FER2013 48×48 grayscale face images and learns spatial filters over facial features. Training uses FER+ unique top-vote consensus labels (minimum four votes), with horizontal flips applied only to training batches. PublicTest selected the model; PrivateTest was evaluated once after selection.", "",
        f"Architecture: {metadata['architecture']}", "",
        "| Model | PublicTest consensus accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| Existing FER+ MLP | {val['baseline_current_model']['accuracy']:.4f} | {val['baseline_current_model']['macro_f1']:.4f} |",
        f"| Spatial CNN + flip averaging | {val['cnn_candidate']['accuracy']:.4f} | {val['cnn_candidate']['macro_f1']:.4f} |", "",
        f"Training stopped after {len(history)} epochs ({val['training_seconds']:.1f}s on {val['device']}).",
        f"PrivateTest consensus: {test['accuracy']:.4f} accuracy and {test['macro_f1']:.4f} macro-F1 on {int(test['classification_report']['weighted avg']['support']):,} accepted rows.", "",
        "Per-class precision, recall, F1, and the confusion matrix are in `final_test_metrics_ferplus_consensus.json`. The candidate validation and training curves are in `validation_metrics.json` and `training_history.json`.",
    ]
    (run / "model_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    stable_dir = ROOT / "results" / "cnn_ferplus"
    stable_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ferplus_data_audit.json", "validation_metrics.json", "training_history.json",
                 "final_test_metrics_ferplus_consensus.json", "model_comparison_report.md"):
        shutil.copy2(run / name, stable_dir / name)
    print(f"Promoted validated CNN: PublicTest accuracy={val['cnn_candidate']['accuracy']:.4f}, "
          f"PrivateTest accuracy={test['accuracy']:.4f}; run={run}")


if __name__ == "__main__":
    main()

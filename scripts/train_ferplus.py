"""Train/evaluate FER+ majority-vote labels aligned to the local FER2013 images."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES, MODEL_DIR
from src.ml.models import final_metrics
from src.tracking import new_run, plot_confusion, save_json

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
# FER+ order is neutral, happiness, surprise, sadness, anger, disgust, fear, ...
# Reorder the seven target emotion votes to match this project's numeric labels.
VOTE_COLUMNS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def load_targets(csv_path: Path, min_votes: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict]:
    labels = pd.read_csv(csv_path)
    required = {"Usage", "Image name", *VOTE_COLUMNS}
    if not required.issubset(labels.columns):
        raise ValueError(f"FER+ labels are missing columns: {sorted(required - set(labels.columns))}")
    if len(labels) != 35887:
        raise ValueError(f"Expected 35,887 aligned FER+ rows, received {len(labels)}")
    votes = labels[VOTE_COLUMNS].to_numpy(dtype=np.int16)
    max_votes = votes.max(axis=1)
    # Keep a unique top vote count among the seven target classes. Non-target
    # FER+ categories are not outputs in this seven-class classifier.
    unique_winner = (votes == max_votes[:, None]).sum(axis=1) == 1
    accepted = (max_votes >= min_votes) & unique_winner & (votes.sum(axis=1) > 0)
    targets = votes.argmax(axis=1).astype(np.int64)
    usage = labels["Usage"].astype(str).to_numpy()
    mapped = {}
    counts = {}
    for split in ("Training", "PublicTest", "PrivateTest"):
        keep = (usage == split) & accepted
        mapped[split] = (np.flatnonzero(keep), targets[keep], max_votes[keep])
        counts[split] = {"available": int((usage == split).sum()), "accepted": int(keep.sum()),
                         "discarded_ambiguous": int(((usage == split) & ~accepted).sum())}
    return mapped, {"rows": int(len(labels)), "min_votes": min_votes,
                    "vote_columns_in_model_order": VOTE_COLUMNS, "counts": counts,
                    "source_url": FERPLUS_URL}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-votes", type=int, default=4,
                        help="Minimum votes for a unique winner among the seven target classes.")
    args = parser.parse_args()
    csv_path = ROOT / "data" / "raw" / "ferplus" / "fer2013new.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = csv_path.with_suffix(".csv.part")
        request = urllib.request.Request(FERPLUS_URL, headers={"User-Agent": "facial-emotion-recognition/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size < 500_000:
                raise ValueError("Downloaded FER+ annotation file is unexpectedly small")
            temporary.replace(csv_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Downloaded FER+ crowd-vote labels from {FERPLUS_URL}", flush=True)

    label_rows, label_info = load_targets(csv_path, args.min_votes)
    # Alignment is guaranteed only when both official files retain the published row order.
    original = pd.read_csv(ROOT / "data" / "raw" / "fer2013" / "fer2013.csv", usecols=["Usage"])
    if len(original) != len(pd.read_csv(csv_path, usecols=["Usage"])):
        raise ValueError("FER+ annotations and FER2013 images are not row-aligned")
    for split in ("Training", "PublicTest", "PrivateTest"):
        split_rows = np.flatnonzero(original["Usage"].astype(str).to_numpy() == split)
        if len(split_rows) != len(np.load(ROOT / "data" / "processed" / f"X_{ {'Training':'train','PublicTest':'val','PrivateTest':'test'}[split] }.npy", mmap_mode="r")):
            raise ValueError(f"Cached {split} images do not match the FER+ row count")
        # The cached arrays follow the original CSV's order within each split.
        label_rows[split] = (np.arange(len(split_rows))[np.isin(split_rows, label_rows[split][0])],
                             label_rows[split][1], label_rows[split][2])

    arrays = {key: np.load(ROOT / "data" / "processed" / f"X_{key}.npy")
              for key in ("train", "val", "test")}
    split_key = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    ix_train, y_train, _ = label_rows["Training"]
    ix_val, y_val, vote_strength_val = label_rows["PublicTest"]
    X_train = arrays["train"][ix_train]
    X_val = arrays["val"][ix_val]

    baseline_bundle = joblib.load(MODEL_DIR / "selected_model.joblib")
    baseline_model = baseline_bundle["model"]
    baseline_strategy = baseline_bundle.get("inference_strategy") == "horizontal_flip_probability_average"
    baseline_validation = final_metrics(baseline_model, X_val, y_val, flip_average=baseline_strategy)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                          alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                          max_iter=60, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=12, random_state=42, verbose=False)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    candidate_validation = final_metrics(model, X_val, y_val, flip_average=True)
    selected = candidate_validation["accuracy"] > baseline_validation["accuracy"]

    run = new_run("ferplus_majority")
    save_json(run / "ferplus_data_audit.json", label_info)
    save_json(run / "validation_baseline_on_ferplus_consensus.json", baseline_validation)
    save_json(run / "validation_candidate_on_ferplus_consensus.json", candidate_validation)
    np.savez_compressed(run / "validation_ferplus_consensus.npz",
                        source_row_index=label_rows["PublicTest"][0],
                        max_votes=vote_strength_val, y=y_val)
    joblib.dump(model, run / "candidate_model.joblib", compress=3)

    report_lines = [
        "# FER+ label quality experiment", "",
        f"FER+ provides crowd-sourced votes aligned row-for-row to the FER2013 CSV. Rows were retained only when one of the seven target expressions had a unique highest vote count of at least {args.min_votes}. Tied, low-vote, and no-target-vote rows were excluded. Non-target labels (contempt, unknown, not-a-face) are not classifier outputs.", "",
        "| Split | FER2013 rows | Accepted FER+ consensus rows | Excluded ambiguous rows |",
        "|---|---:|---:|---:|"
    ]
    for split, counts in label_info["counts"].items():
        report_lines.append(f"| {split} | {counts['available']:,} | {counts['accepted']:,} | {counts['discarded_ambiguous']:,} |")
    report_lines.extend(["", "Validation uses the PublicTest partition with FER+ consensus labels. The PrivateTest partition is untouched until validation selects the candidate.", "",
                         "| Validation model | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    report_lines.append(f"| Existing selected MLP + its current inference strategy | {baseline_validation['accuracy']:.4f} | {baseline_validation['macro_f1']:.4f} |")
    report_lines.append(f"| MLP trained on FER+ consensus labels + flip averaging | {candidate_validation['accuracy']:.4f} | {candidate_validation['macro_f1']:.4f} |")

    candidate_bundle = {"model": model, "name": "mlp_128_64_ferplus_majority",
                        "class_names": list(CLASS_NAMES), "mode": "full_ferplus",
                        "inference_strategy": "horizontal_flip_probability_average",
                        "validation_accuracy": candidate_validation["accuracy"],
                        "validation_macro_f1": candidate_validation["macro_f1"],
                        "inference_comparison_run": str(run.relative_to(ROOT))}
    if selected:
        ix_test, y_test, _ = label_rows["PrivateTest"]
        X_test = arrays["test"][ix_test]
        test_metrics = final_metrics(model, X_test, y_test, flip_average=True)
        save_json(run / "final_test_metrics_ferplus_consensus.json", test_metrics)
        plot_confusion(test_metrics["confusion_matrix"], "FER+ majority MLP · PrivateTest consensus",
                       run / "final_test_confusion_matrix.png")
        report_lines.extend(["", f"Candidate selected by validation accuracy. PrivateTest was evaluated once after selection on {len(y_test):,} FER+ consensus rows: accuracy **{test_metrics['accuracy']:.4f}**, macro-F1 **{test_metrics['macro_f1']:.4f}**."])

        current_model = MODEL_DIR / "selected_model.joblib"
        backup_model = MODEL_DIR / "selected_model_before_ferplus.joblib"
        backup_metadata = MODEL_DIR / "selected_model_before_ferplus.json"
        if not backup_model.exists():
            shutil.copy2(current_model, backup_model)
        current_metadata = MODEL_DIR / "selected_model.json"
        if current_metadata.exists() and not backup_metadata.exists():
            shutil.copy2(current_metadata, backup_metadata)
        candidate_path = MODEL_DIR / "selected_model_ferplus.joblib"
        joblib.dump(candidate_bundle, candidate_path, compress=3)
        shutil.copy2(candidate_path, current_model)
        metadata = {key: value for key, value in candidate_bundle.items() if key != "model"}
        metadata["test_metrics_ferplus_consensus"] = test_metrics
        metadata["baseline_validation_ferplus_consensus"] = baseline_validation
        metadata["data_minimum_votes"] = args.min_votes
        (MODEL_DIR / "selected_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        report_lines.extend(["", "The FER+ candidate did not exceed the current model on validation accuracy, so it was not promoted and the private test partition was not evaluated."])

    (run / "model_comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Training examples: {len(y_train):,}; validation examples: {len(y_val):,}")
    print(f"Validation baseline on FER+ consensus: accuracy={baseline_validation['accuracy']:.4f}, macro_f1={baseline_validation['macro_f1']:.4f}")
    print(f"Validation FER+ model: accuracy={candidate_validation['accuracy']:.4f}, macro_f1={candidate_validation['macro_f1']:.4f}, fit_seconds={train_seconds:.1f}")
    print(f"Promoted: {selected}; run: {run}")
    if selected:
        print(f"PrivateTest FER+ consensus: accuracy={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

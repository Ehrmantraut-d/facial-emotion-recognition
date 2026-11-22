"""Immutable experiment run folders and generated scientific plots."""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import CLASS_NAMES, RESULTS_DIR, RUNS_DIR

LOG = logging.getLogger(__name__)

def new_run(mode: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{stamp}_{mode}_{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path

def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def plot_confusion(matrix: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.asarray(matrix), annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_comparison(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame([{k: r[k] for k in ("name", "accuracy", "macro_f1", "train_seconds", "prediction_latency_ms_per_sample", "train_samples")} for r in rows])
    df.to_csv(out / "validation_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4)); df.plot(x="name", y=["accuracy", "macro_f1"], kind="bar", ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("Validation score"); ax.set_title("Validation model comparison"); fig.tight_layout()
    fig.savefig(out / "validation_comparison.png", dpi=150); plt.close(fig)

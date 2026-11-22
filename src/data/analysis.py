"""Generate descriptive tables and portable plots from the acquired dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASS_NAMES, RESULTS_DIR


def _font(size: int = 18) -> ImageFont.ImageFont:
    """Use a bundled system font when available, with Pillow's fallback."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_class_distribution(counts: list[int], path: Path) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font = _font(26), _font(18)
    draw.text((42, 22), "Training samples per expression", fill="#172033", font=title_font)
    left, top, right, bottom = 90, 92, 1160, 450
    maximum = max(counts) or 1
    for tick in range(5):
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#e5eaf2", width=2)
        draw.text((12, y - 10), f"{maximum * tick // 4:,}", fill="#667085", font=label_font)
    slot = (right - left) / len(CLASS_NAMES)
    for index, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        x0 = left + index * slot + 22
        x1 = left + (index + 1) * slot - 22
        y1 = bottom
        y0 = bottom - (bottom - top) * count / maximum
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill="#4f6bed")
        draw.text((x0, y0 - 25), f"{count:,}", fill="#344054", font=label_font)
        draw.text((x0, bottom + 18), name, fill="#344054", font=label_font)
    image.save(path)


def _save_pixel_distribution(values: np.ndarray, path: Path) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font = _font(26), _font(18)
    draw.text((42, 22), "Normalized pixel intensity distribution", fill="#172033", font=title_font)
    left, top, right, bottom = 90, 92, 1160, 450
    counts, _ = np.histogram(values, bins=40, range=(0, 1))
    maximum = max(int(counts.max()), 1)
    for tick in range(5):
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#e5eaf2", width=2)
    slot = (right - left) / len(counts)
    for index, count in enumerate(counts):
        x0 = left + index * slot + 1
        x1 = left + (index + 1) * slot - 1
        y0 = bottom - (bottom - top) * int(count) / maximum
        draw.rectangle((x0, y0, x1, bottom), fill="#7666d8")
    draw.text((left, bottom + 18), "0.0", fill="#344054", font=label_font)
    draw.text((right - 35, bottom + 18), "1.0", fill="#344054", font=label_font)
    draw.text((left, bottom + 48), "Pixel value [0, 1]", fill="#667085", font=label_font)
    image.save(path)


def _save_class_examples(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    tile, padding, label_height = 144, 18, 46
    width = padding + len(CLASS_NAMES) * (tile + padding)
    height = tile + label_height + padding * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_font = _font(17)
    for index, name in enumerate(CLASS_NAMES):
        x = padding + index * (tile + padding)
        matches = np.flatnonzero(y == index)
        if len(matches):
            pixels = np.clip(X[matches[0]].reshape(48, 48) * 255, 0, 255).astype(np.uint8)
            face = Image.fromarray(pixels, mode="L").resize((tile, tile), Image.Resampling.NEAREST)
            image.paste(face.convert("RGB"), (x, padding))
        draw.text((x, padding + tile + 12), name, fill="#344054", font=label_font)
    image.save(path)


def generate_eda(data: dict, validation: dict, output_dir: Path | None = None) -> Path:
    out = output_dir or RESULTS_DIR / "dataset_analysis"
    graphs = out / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)

    X, y = data["train"]
    counts = [int(np.sum(y == index)) for index in range(len(CLASS_NAMES))]
    distribution = pd.DataFrame({"class": CLASS_NAMES, "samples": counts})
    distribution["percent"] = distribution.samples / max(distribution.samples.sum(), 1) * 100
    distribution.to_csv(out / "class_distribution.csv", index=False)
    stats = {
        "prepared_split_counts": {key: int(len(value[1])) for key, value in data.items()},
        "training_class_counts": dict(zip(CLASS_NAMES, distribution.samples.tolist())),
        "training_class_percentages": dict(zip(CLASS_NAMES, distribution.percent.round(4).tolist())),
        "normalized_pixel_min": float(X.min()),
        "normalized_pixel_max": float(X.max()),
        "normalized_pixel_mean": float(X.mean()),
        "normalized_pixel_std": float(X.std()),
        "invalid_samples": int(validation.get("invalid_count", 0)),
        "dimensions": [48, 48],
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _save_class_distribution(counts, graphs / "class_distribution.png")
    _save_pixel_distribution(X, graphs / "pixel_distribution.png")
    _save_class_examples(X, y, graphs / "class_examples.png")
    return out

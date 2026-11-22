"""Central project paths and FER2013 class definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "fer2013"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
MODEL_DIR = ROOT / "models"
ENVIRONMENT_DIR = RESULTS_DIR / "environment"
CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
FRENCH_NAMES = ("Colère", "Dégoût", "Peur", "Joie", "Tristesse", "Surprise", "Neutre")
IMAGE_SHAPE = (48, 48)
RANDOM_SEED = 42

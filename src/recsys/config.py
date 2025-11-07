"""Configuration helpers for project paths and defaults."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ARCHIVE_NAME = "ml-latest-small.zip"
MOVIELENS_DIR_NAME = "ml-latest-small"

DEFAULT_MIN_RATINGS_PER_USER = 5
DEFAULT_TEST_RATIO = 0.2
DEFAULT_MIN_RATING = 0.0
DEFAULT_TOP_K = 10


def ensure_directories() -> None:
    """Ensure that the project directories exist."""
    for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

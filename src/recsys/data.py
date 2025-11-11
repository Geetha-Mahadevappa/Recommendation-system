"""Utilities for downloading and preparing the MovieLens dataset."""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from lightfm.data import Dataset
from scipy import sparse
from tqdm import tqdm

from . import config

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Container for all artefacts required to train and evaluate the model."""

    dataset: Dataset
    train_interactions: sparse.coo_matrix
    test_interactions: sparse.coo_matrix
    item_features: sparse.csr_matrix
    metadata: Dict[str, object]


def _download_file(url: str, destination: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(destination, "wb") as file, tqdm(
        total=total, unit="B", unit_scale=True, desc=destination.name
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress.update(len(chunk))


def download_movielens(force: bool = False) -> Path:
    """Download and extract the MovieLens dataset."""
    dataset_dir = config.DATA_DIR / "movielens"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dataset_dir / "ml-latest-small.zip"
    extracted_dir = dataset_dir / "ml-latest-small"

    if extracted_dir.exists() and not force:
        LOGGER.info("MovieLens dataset already extracted.")
        return extracted_dir

    url = config.MOVIELENS_URL
    LOGGER.info("Downloading MovieLens dataset from %s", url)
    _download_file(url, archive_path)

    LOGGER.info("Extracting MovieLens dataset...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    LOGGER.info("Dataset ready at %s", extracted_dir)
    return extracted_dir


def _load_ratings(dataset_dir: Path) -> pd.DataFrame:
    ratings_path = dataset_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(
            "ratings.csv was not found. Have you downloaded the dataset?"
        )
    ratings = pd.read_csv(ratings_path)
    return ratings


def _filter_users(
    ratings: pd.DataFrame,
    min_ratings_per_user: int,
    min_rating: float,
) -> pd.DataFrame:
    ratings = ratings[ratings["rating"] >= min_rating]
    counts = ratings.groupby("userId")["movieId"].transform("count")
    filtered = ratings[counts >= min_ratings_per_user]
    return filtered


def _train_test_split(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp"])
    test = ratings.groupby("userId").tail(1)
    train = ratings.drop(test.index)
    return train, test


def _extract_genre_features(
    movies: pd.DataFrame,
) -> Tuple[List[str], Iterable[Tuple[int, List[str]]]]:
    all_genres = set()
    item_features = []
    for row in movies.itertuples(index=False):
        genres = []
        if isinstance(row.genres, str):
            for genre in row.genres.split("|"):
                genre = genre.strip()
                if genre and genre.lower() != "(no genres listed)":
                    feature = f"genre:{genre.lower()}"
                    genres.append(feature)
                    all_genres.add(feature)
        item_features.append((row.movieId, genres))
    return sorted(all_genres), item_features


def _summarise_dataset(
    filtered: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
) -> Dict[str, object]:
    """Create high-level summary statistics for the prepared dataset."""

    num_users = int(filtered["userId"].nunique())
    num_items = int(filtered["movieId"].nunique())

    rating_stats = filtered["rating"].agg(["min", "max", "mean", "median"])  # type: ignore[call-overload]
    rating_summary = {
        "min": float(rating_stats["min"]),
        "max": float(rating_stats["max"]),
        "mean": float(rating_stats["mean"]),
        "median": float(rating_stats["median"]),
    }

    merged = filtered.merge(
        movies[["movieId", "title", "genres"]], on="movieId", how="left"
    )

    genre_counter: Counter[str] = Counter()
    for genres in merged["genres"].dropna():
        for genre in str(genres).split("|"):
            genre = genre.strip()
            if genre and genre.lower() != "(no genres listed)":
                genre_counter[genre] += 1

    top_genres = [
        {"genre": genre, "count": int(count)}
        for genre, count in genre_counter.most_common(5)
    ]

    movie_counts = (
        merged.groupby(["movieId", "title"], dropna=False)["rating"]
        .count()
        .sort_values(ascending=False)
        .head(5)
    )

    top_movies = [
        {
            "movie_id": int(movie_id),
            "title": title if isinstance(title, str) else "Unknown",
            "ratings": int(count),
        }
        for (movie_id, title), count in movie_counts.items()
    ]

    total_interactions = int(filtered.shape[0])
    train_interactions = int(train.shape[0])
    test_interactions = int(test.shape[0])

    sparsity = 1.0 - (
        train_interactions / float(max(num_users * num_items, 1))
    )

    return {
        "users": num_users,
        "items": num_items,
        "interactions": {
            "total": total_interactions,
            "train": train_interactions,
            "test": test_interactions,
        },
        "rating_distribution": rating_summary,
        "sparsity": float(sparsity),
        "top_genres": top_genres,
        "top_movies": top_movies,
    }


def _build_dataset(
    filtered: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
) -> PreparedData:
    dataset = Dataset()

    genres, item_feature_tuples = _extract_genre_features(movies)
    dataset.fit(
        users=train["userId"].unique(),
        items=train["movieId"].unique(),
        item_features=genres,
    )

    item_features = dataset.build_item_features(item_feature_tuples).tocsr()

    def _build_interactions(frame: pd.DataFrame) -> sparse.coo_matrix:
        interactions, _ = dataset.build_interactions(
            (row.userId, row.movieId, float(row.rating))
            for row in frame.itertuples(index=False)
        )
        return interactions.tocoo()

    train_interactions = _build_interactions(train)
    test_interactions = _build_interactions(test)

    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

    metadata = {
        "user_id_map": {str(key): int(value) for key, value in user_id_map.items()},
        "item_id_map": {str(key): int(value) for key, value in item_id_map.items()},
        "id_to_user": {str(int(value)): str(key) for key, value in user_id_map.items()},
        "id_to_item": {str(int(value)): int(key) for key, value in item_id_map.items()},
        "item_feature_map": {
            str(key): value for key, value in item_feature_map.items()
        },
        "movie_titles": {
            str(int(row.movieId)): row.title for row in movies.itertuples(index=False)
        },
        "movies": movies.to_dict(orient="records"),
        "summary": _summarise_dataset(filtered, train, test, movies),
    }

    return PreparedData(
        dataset=dataset,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        item_features=item_features,
        metadata=metadata,
    )


def prepare_dataset(
    min_ratings_per_user: int = config.DEFAULT_MIN_RATINGS_PER_USER,
    min_rating: float = config.DEFAULT_MIN_RATING,
    force_download: bool = False,
) -> PreparedData:
    """Prepare the MovieLens dataset for training."""
    dataset_dir = download_movielens(force=force_download)

    ratings = _load_ratings(dataset_dir)
    movies = pd.read_csv(dataset_dir / "movies.csv")

    filtered = _filter_users(ratings, min_ratings_per_user, min_rating)
    train, test = _train_test_split(filtered)

    prepared = _build_dataset(filtered, train, test, movies)

    _persist_prepared_data(prepared)
    return prepared


def _persist_prepared_data(prepared: PreparedData) -> None:
    config.ensure_directories()
    sparse.save_npz(config.PROCESSED_DATA_DIR / "train_interactions.npz", prepared.train_interactions)
    sparse.save_npz(config.PROCESSED_DATA_DIR / "test_interactions.npz", prepared.test_interactions)
    sparse.save_npz(config.PROCESSED_DATA_DIR / "item_features.npz", prepared.item_features)

    metadata_path = config.PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(prepared.metadata, file, indent=2)


def load_prepared_data() -> PreparedData:
    """Load prepared artefacts from disk."""
    config.ensure_directories()
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run the 'prepare' command first."
        )

    train_interactions = sparse.load_npz(config.PROCESSED_DATA_DIR / "train_interactions.npz")
    test_interactions = sparse.load_npz(config.PROCESSED_DATA_DIR / "test_interactions.npz")
    item_features = sparse.load_npz(config.PROCESSED_DATA_DIR / "item_features.npz")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    dataset = Dataset()
    return PreparedData(
        dataset=dataset,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        item_features=item_features,
        metadata=metadata,
    )

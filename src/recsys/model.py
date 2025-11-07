"""Model training, evaluation, and inference helpers."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from scipy import sparse

from . import config, data

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    num_components: int = 64
    learning_rate: float = 0.05
    loss: str = "warp"
    epochs: int = 30
    num_threads: int = 4
    random_state: int = 42


def train_model(
    prepared: data.PreparedData,
    training_config: TrainingConfig | None = None,
) -> LightFM:
    """Train a LightFM model using the prepared dataset."""
    if training_config is None:
        training_config = TrainingConfig()

    model = LightFM(
        no_components=training_config.num_components,
        learning_rate=training_config.learning_rate,
        loss=training_config.loss,
        random_state=training_config.random_state,
    )

    LOGGER.info(
        "Training LightFM model (loss=%s, components=%d, epochs=%d)",
        training_config.loss,
        training_config.num_components,
        training_config.epochs,
    )

    model.fit(
        prepared.train_interactions,
        item_features=prepared.item_features,
        epochs=training_config.epochs,
        num_threads=training_config.num_threads,
        verbose=True,
    )

    return model


@dataclass
class EvaluationResults:
    precision_at_k: float
    recall_at_k: float
    auc: float


def evaluate_model(
    model: LightFM,
    prepared: data.PreparedData,
    k: int = config.DEFAULT_TOP_K,
    num_threads: int = 4,
) -> EvaluationResults:
    """Evaluate the trained model on precision@k, recall@k and AUC."""
    precision = precision_at_k(
        model,
        prepared.test_interactions,
        train_interactions=prepared.train_interactions,
        item_features=prepared.item_features,
        k=k,
        num_threads=num_threads,
    ).mean()

    recall = recall_at_k(
        model,
        prepared.test_interactions,
        train_interactions=prepared.train_interactions,
        item_features=prepared.item_features,
        k=k,
        num_threads=num_threads,
    ).mean()

    auc = auc_score(
        model,
        prepared.test_interactions,
        train_interactions=prepared.train_interactions,
        item_features=prepared.item_features,
        num_threads=num_threads,
    ).mean()

    return EvaluationResults(
        precision_at_k=float(precision),
        recall_at_k=float(recall),
        auc=float(auc),
    )


def _get_user_row(interactions: sparse.coo_matrix, internal_user_id: int) -> sparse.csr_matrix:
    matrix = interactions.tocsr()
    return matrix[internal_user_id]


def recommend_for_user(
    model: LightFM,
    prepared: data.PreparedData,
    user_id: int,
    k: int = config.DEFAULT_TOP_K,
) -> List[Dict[str, object]]:
    metadata = prepared.metadata
    user_id_map = metadata["user_id_map"]
    if str(user_id) not in user_id_map:
        raise KeyError(
            f"User {user_id} is not part of the training data."
        )

    internal_user_id = int(user_id_map[str(user_id)])
    item_ids = np.array(sorted(int(key) for key in metadata["id_to_item"].keys()))

    scores = model.predict(
        user_ids=internal_user_id,
        item_ids=item_ids,
        item_features=prepared.item_features,
    )

    known_items = set(_get_user_row(prepared.train_interactions, internal_user_id).indices)

    ranked_items = [
        (internal_item_id, score)
        for internal_item_id, score in zip(item_ids, scores)
        if internal_item_id not in known_items
    ]
    ranked_items.sort(key=lambda pair: pair[1], reverse=True)
    top_items = ranked_items[:k]

    results: List[Dict[str, object]] = []
    for internal_item_id, score in top_items:
        movie_id = metadata["id_to_item"][str(int(internal_item_id))]
        title = metadata["movie_titles"].get(str(int(movie_id)))
        if title is None:
            title = f"Movie {movie_id}"
        results.append(
            {
                "movie_id": int(movie_id),
                "title": title,
                "score": float(score),
            }
        )
    return results


def save_model(model: LightFM, path: Path | None = None) -> Path:
    config.ensure_directories()
    if path is None:
        path = config.MODELS_DIR / "lightfm_model.pkl"
    with open(path, "wb") as file:
        pickle.dump(model, file)
    LOGGER.info("Saved model to %s", path)
    return path


def load_model(path: Path | None = None) -> LightFM:
    if path is None:
        path = config.MODELS_DIR / "lightfm_model.pkl"
    if not path.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def export_recommendations(
    recommendations: Sequence[Dict[str, object]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as file:
        json.dump(list(recommendations), file, indent=2)
    LOGGER.info("Wrote recommendations to %s", destination)

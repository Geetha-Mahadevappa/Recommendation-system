"""Command-line interface for the MovieLens recommendation system."""
from __future__ import annotations

import json
from typing import Optional

import typer

from . import config, data, model

DEFAULT_TRAINING_CONFIG = model.TrainingConfig()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Utilities for preparing data, training models, and generating recommendations.",
)


def _echo_summary(summary: dict[str, object]) -> None:
    """Pretty-print a prepared dataset summary."""
    interactions = summary.get("interactions", {}) if isinstance(summary, dict) else {}
    ratings = summary.get("rating_distribution", {}) if isinstance(summary, dict) else {}

    typer.echo("Dataset summary:")
    typer.echo(f"  Users: {summary.get('users', 'unknown')}")
    typer.echo(f"  Items: {summary.get('items', 'unknown')}")
    typer.echo("  Interactions:")
    typer.echo(f"    Total: {interactions.get('total', 'unknown')}")
    typer.echo(f"    Train: {interactions.get('train', 'unknown')}")
    typer.echo(f"    Test: {interactions.get('test', 'unknown')}")
    typer.echo("  Rating distribution:")
    typer.echo(
        "    Min={min:.2f} Max={max:.2f} Mean={mean:.2f} Median={median:.2f}".format(
            min=ratings.get("min", 0.0),
            max=ratings.get("max", 0.0),
            mean=ratings.get("mean", 0.0),
            median=ratings.get("median", 0.0),
        )
    )
    sparsity = summary.get("sparsity")
    if isinstance(sparsity, float):
        typer.echo(f"  Matrix sparsity: {sparsity:.4f}")

    top_genres = summary.get("top_genres") or []
    if isinstance(top_genres, list) and top_genres:
        typer.echo("  Top genres:")
        for genre in top_genres:
            typer.echo(f"    {genre.get('genre')}: {genre.get('count')}")

    top_movies = summary.get("top_movies") or []
    if isinstance(top_movies, list) and top_movies:
        typer.echo("  Most-rated movies:")
        for movie in top_movies:
            typer.echo(
                "    {title} (movie_id={movie_id}) – ratings={ratings}".format(
                    title=movie.get("title"),
                    movie_id=movie.get("movie_id"),
                    ratings=movie.get("ratings"),
                )
            )


@app.command()
def download(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download and re-extract the MovieLens dataset even if it already exists.",
    ),
) -> None:
    """Download the MovieLens dataset archive and extract it locally."""
    dataset_dir = data.download_movielens(force=force)
    typer.echo(f"MovieLens data available at {dataset_dir}")


@app.command()
def prepare(
    min_ratings_per_user: int = typer.Option(
        config.DEFAULT_MIN_RATINGS_PER_USER,
        help="Discard users with fewer than this many ratings.",
    ),
    min_rating: float = typer.Option(
        config.DEFAULT_MIN_RATING,
        help="Drop ratings below this threshold before training.",
    ),
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Force a fresh copy of the MovieLens archive before preparing data.",
    ),
) -> None:
    """Prepare the MovieLens dataset and persist the processed artefacts."""
    prepared = data.prepare_dataset(
        min_ratings_per_user=min_ratings_per_user,
        min_rating=min_rating,
        force_download=force_download,
    )
    typer.echo("Prepared dataset saved to disk.")
    summary = prepared.metadata.get("summary")
    if isinstance(summary, dict):
        _echo_summary(summary)


@app.command()
def describe(
    as_json: bool = typer.Option(False, help="Emit dataset summary as JSON."),
) -> None:
    """Display descriptive statistics for the processed dataset."""
    prepared = data.load_prepared_data()
    summary = prepared.metadata.get("summary")
    if summary is None:
        typer.echo(
            "Summary statistics are unavailable. Prepare the dataset to generate them.",
            err=True,
        )
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return

    _echo_summary(summary)


@app.command()
def train(
    loss: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.loss,
        help="LightFM loss function (e.g. warp, bpr, logistic).",
    ),
    num_components: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.num_components,
        help="Dimensionality of the latent representation.",
    ),
    learning_rate: float = typer.Option(
        DEFAULT_TRAINING_CONFIG.learning_rate,
        help="LightFM learning rate.",
    ),
    epochs: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.epochs,
        help="Number of training epochs.",
    ),
    num_threads: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.num_threads,
        help="Number of threads to use when fitting the model.",
    ),
    random_state: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.random_state,
        help="Random seed for reproducibility.",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        help="Optional destination path for the trained model pickle.",
        dir_okay=True,
        file_okay=True,
    ),
    evaluate_after_training: bool = typer.Option(
        True,
        "--evaluate/--no-evaluate",
        help="Evaluate the model on the test set after training completes.",
    ),
) -> None:
    """Train a LightFM model on the prepared dataset."""
    prepared = data.load_prepared_data()
    training_config = model.TrainingConfig(
        num_components=num_components,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        num_threads=num_threads,
        random_state=random_state,
    )
    trained_model = model.train_model(prepared, training_config=training_config)

    saved_path = model.save_model(trained_model, path=model_path)
    typer.echo(f"Model saved to {saved_path}")

    if evaluate_after_training:
        results = model.evaluate_model(
            trained_model,
            prepared,
            k=config.DEFAULT_TOP_K,
            num_threads=num_threads,
        )
        typer.echo("Evaluation metrics:")
        typer.echo(f"  Precision@{config.DEFAULT_TOP_K}: {results.precision_at_k:.4f}")
        typer.echo(f"  Recall@{config.DEFAULT_TOP_K}: {results.recall_at_k:.4f}")
        typer.echo(f"  AUC: {results.auc:.4f}")


@app.command()
def evaluate(
    model_path: Optional[Path] = typer.Option(
        None,
        help="Path to a trained LightFM model (defaults to the last saved model).",
    ),
    k: int = typer.Option(
        config.DEFAULT_TOP_K,
        help="Evaluate recall@k and precision@k at this cut-off.",
    ),
    num_threads: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.num_threads,
        help="Number of threads used for evaluation metrics.",
    ),
) -> None:
    """Evaluate a trained model using the prepared test interactions."""
    prepared = data.load_prepared_data()
    trained_model = model.load_model(path=model_path)
    results = model.evaluate_model(
        trained_model,
        prepared,
        k=k,
        num_threads=num_threads,
    )
    typer.echo("Evaluation metrics:")
    typer.echo(f"  Precision@{k}: {results.precision_at_k:.4f}")
    typer.echo(f"  Recall@{k}: {results.recall_at_k:.4f}")
    typer.echo(f"  AUC: {results.auc:.4f}")


@app.command()
def recommend(
    user_id: int = typer.Argument(..., help="External MovieLens user identifier."),
    k: int = typer.Option(
        config.DEFAULT_TOP_K,
        help="Number of recommendations to generate.",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        help="Path to the trained LightFM model to use for inference.",
    ),
    export: Optional[Path] = typer.Option(
        None,
        help="Optional path to export the recommendations as JSON.",
    ),
) -> None:
    """Generate top-N movie recommendations for a given user."""
    prepared = data.load_prepared_data()
    trained_model = model.load_model(path=model_path)

    recommendations = model.recommend_for_user(
        trained_model,
        prepared,
        user_id=user_id,
        k=k,
    )

    if not recommendations:
        typer.echo("No recommendations available for this user.")
        return

    typer.echo(f"Top {len(recommendations)} recommendations for user {user_id}:")
    for index, item in enumerate(recommendations, start=1):
        typer.echo(
            "  {idx}. {title} (movie_id={movie_id}) – score={score:.4f}".format(
                idx=index,
                title=item.get("title"),
                movie_id=item.get("movie_id"),
                score=item.get("score", 0.0),
            )
        )

    if export is not None:
        model.export_recommendations(recommendations, export)
        typer.echo(f"Recommendations exported to {export}")


if __name__ == "__main__":  # pragma: no cover - entrypoint convenience
    app()

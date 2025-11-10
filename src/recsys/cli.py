"""Command line interface for the recommendation system project."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from . import config, data, model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="MovieLens recommendation system reference project")


@app.command()
def download(force: bool = typer.Option(False, help="Redownload the dataset even if it exists.")) -> None:
    """Download the MovieLens dataset."""
    path = data.download_movielens(force=force)
    typer.echo(f"Dataset available at {path}")


@app.command()
def prepare(
    min_ratings_per_user: int = typer.Option(
        config.DEFAULT_MIN_RATINGS_PER_USER,
        help="Minimum number of ratings per user to keep.",
    ),
    min_rating: float = typer.Option(
        config.DEFAULT_MIN_RATING,
        help="Minimum rating value to keep",
    ),
    force_download: bool = typer.Option(False, help="Force re-downloading the dataset."),
) -> None:
    """Prepare the dataset and persist the processed artefacts."""
    prepared = data.prepare_dataset(
        min_ratings_per_user=min_ratings_per_user,
        min_rating=min_rating,
        force_download=force_download,
    )
    typer.echo(
        "Prepared dataset with "
        f"{prepared.train_interactions.shape[0]} users and "
        f"{prepared.train_interactions.shape[1]} items."
    )


@app.command()
def train(
    num_components: int = typer.Option(64, help="Number of latent factors."),
    learning_rate: float = typer.Option(0.05, help="Learning rate for the optimizer."),
    loss: str = typer.Option("warp", help="Loss function (warp, bpr, logistic)."),
    epochs: int = typer.Option(30, help="Number of training epochs."),
    num_threads: int = typer.Option(4, help="Number of CPU threads to use."),
) -> None:
    """Train the LightFM model and save it to disk."""
    prepared = data.load_prepared_data()
    training_config = model.TrainingConfig(
        num_components=num_components,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        num_threads=num_threads,
    )

    trained_model = model.train_model(prepared, training_config)
    model_path = model.save_model(trained_model)
    results = model.evaluate_model(trained_model, prepared)

    typer.echo(f"Model saved to {model_path}")
    typer.echo(
        json.dumps(
            {
                "precision_at_k": results.precision_at_k,
                "recall_at_k": results.recall_at_k,
                "auc": results.auc,
            },
            indent=2,
        )
    )


@app.command()
def evaluate(k: int = typer.Option(config.DEFAULT_TOP_K, help="Cutoff for precision/recall.")) -> None:
    """Evaluate the most recently trained model."""
    prepared = data.load_prepared_data()
    trained_model = model.load_model()
    results = model.evaluate_model(trained_model, prepared, k=k)
    typer.echo(
        json.dumps(
            {
                "precision_at_k": results.precision_at_k,
                "recall_at_k": results.recall_at_k,
                "auc": results.auc,
            },
            indent=2,
        )
    )


@app.command()
def describe(as_json: bool = typer.Option(False, help="Emit dataset summary as JSON.")) -> None:
    """Show descriptive statistics for the processed dataset."""
    prepared = data.load_prepared_data()
    summary = prepared.metadata.get("summary")
    if summary is None:
        typer.echo(
            "Summary statistics are unavailable. Prepare the dataset to generate them."
        )
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return

    interactions = summary.get("interactions", {})
    ratings = summary.get("rating_distribution", {})

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
    if top_genres:
        typer.echo("  Top genres:")
        for genre in top_genres:
            typer.echo(f"    {genre.get('genre')}: {genre.get('count')}")

    top_movies = summary.get("top_movies") or []
    if top_movies:
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
def recommend(
    user_id: int = typer.Argument(..., help="User ID from the dataset."),
    k: int = typer.Option(config.DEFAULT_TOP_K, help="Number of recommendations to return."),
    export: Optional[Path] = typer.Option(None, help="Optional path to export recommendations as JSON."),
) -> None:
    """Generate personalised movie recommendations for a user."""
    prepared = data.load_prepared_data()
    trained_model = model.load_model()
    recommendations = model.recommend_for_user(trained_model, prepared, user_id=user_id, k=k)

    if not recommendations:
        typer.echo("No recommendations available. Try training the model first.")
        raise typer.Exit(code=1)

    if export:
        model.export_recommendations(recommendations, export)
        typer.echo(f"Recommendations exported to {export}")
    else:
        for index, item in enumerate(recommendations, start=1):
            typer.echo(f"{index}. {item['title']} (movie_id={item['movie_id']}) -> score={item['score']:.3f}")


if __name__ == "__main__":
    app()

# Recommendation-system
---

````markdown
# 🎬 MovieLens Recommendation System

This repository contains an end-to-end **production-style recommendation system** built on the [MovieLens](https://grouplens.org/datasets/movielens/) “latest-small” dataset.  
It demonstrates a modern hybrid collaborative filtering approach using the [LightFM](https://making.lyst.com/lightfm/) model with the **WARP loss**, and provides a clean, modular, and reproducible workflow — perfect for learning, portfolio projects, or production prototypes.

---

## 🚀 Features

- **Automated data pipeline** – Download, extract, and preprocess the MovieLens data with a single command.  
- **Hybrid recommender** – Train a LightFM model that blends collaborative filtering with content-based genre features.  
- **Command-line interface (CLI)** – Reproducible commands for preparing data, training, evaluating, and generating personalized recommendations.  
- **Extensible architecture** – Modular Python package, ready for experimentation with different datasets or models.  

---

## 🧩 Getting Started

### 1️⃣ Set up the environment

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"  # Ensures the recsys package is importable
````

---

### 2️⃣ Prepare the dataset

Download and preprocess the MovieLens dataset:

```bash
python -m recsys.cli download   # Downloads the MovieLens latest-small dataset (~1 MB)
python -m recsys.cli prepare    # Filters users, splits train/test, and saves artifacts
```

---

### 3️⃣ Train and evaluate the model

Train the recommender model and evaluate it on key metrics (Precision@k, Recall@k, AUC):

```bash
python -m recsys.cli train
```

You can adjust hyperparameters such as loss type or number of components:

```bash
python -m recsys.cli train --loss bpr --num-components 128
```

To view all options:

```bash
python -m recsys.cli --help
```

---

### 4️⃣ Generate recommendations

Generate personalized recommendations for a specific user:

```bash
python -m recsys.cli recommend 1 --k 5
```

Replace `1` with a valid user ID.
You can also export the recommendations to a JSON file:

```bash
python -m recsys.cli recommend 1 --k 5 --export output.json
```

---

## 📁 Project Structure

```
.
├── requirements.txt          # Python dependencies (LightFM, pandas, Typer, ...)
├── src/
│   └── recsys/
│       ├── cli.py            # Typer-based command-line interface
│       ├── config.py         # Shared paths and configuration constants
│       ├── data.py           # Downloading, filtering, and interaction matrix builders
│       ├── evaluation.py     # Evaluation metrics and helper functions
│       └── model.py          # Training loop, evaluation, persistence, recommendations
├── data/                     # Generated datasets (ignored by Git)
├── models/                   # Persisted LightFM weights (ignored by Git)
└── notebooks/                # Optional notebooks or experiments (ignored by Git)
```

---

## ⚙️ Customization

You can tweak several pipeline parameters easily:

| Option                   | Description                                      | Example                                                     |
| ------------------------ | ------------------------------------------------ | ----------------------------------------------------------- |
| `--min-ratings-per-user` | Filter by minimum user activity                  | `python -m recsys.cli prepare --min-ratings-per-user 10`    |
| `--loss`                 | Choose loss function (`warp`, `bpr`, `logistic`) | `python -m recsys.cli train --loss bpr`                     |
| `--num-components`       | Set latent dimensionality                        | `python -m recsys.cli train --num-components 128`           |
| `--export`               | Export recommendations to JSON                   | `python -m recsys.cli recommend 1 --k 5 --export recs.json` |

---

## 🚧 Next Steps

To extend or productionize this project:

* 📊 Add offline evaluation dashboards (e.g., Jupyter or Streamlit).
* 🧠 Track experiments with **MLflow** or **Weights & Biases**.
* ⚡ Serve the trained model with **FastAPI** and expose a REST endpoint.
* 🎥 Explore sequence-aware or contextual recommenders for richer insights.

---

## 🏁 License

This project is released under the [MIT License](LICENSE).
Feel free to use, modify, and share it for learning or real-world applications.

```markdown
# MovieLens Recommendation System Project

This repository contains an end-to-end example of a production-style recommendation
system built on top of the [MovieLens](https://grouplens.org/datasets/movielens/)
"latest-small" dataset. It showcases a modern, well-known collaborative filtering
approach (the [LightFM](https://making.lyst.com/lightfm/) hybrid model with the
WARP loss) and provides a fully scripted workflow that you can adapt for job-ready
projects or portfolio pieces.

## Features

- **Automated data pipeline** – download, extract, and preprocess the MovieLens data
  with a single command.
- **Hybrid recommender** – train a LightFM model that blends collaborative filtering
  signals with content-based genre features.
- **Command-line interface** – reproducible commands for preparing data, training,
  evaluating, and generating personalised recommendations.
- **Extensible code structure** – modular Python package ready for experimentation
  with different models or datasets.

## Getting started

### 1. Set up the environment

First, create a Python virtual environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"  # Ensures the recsys package is importable
```

### 2. Prepare the dataset

Download and preprocess the MovieLens dataset using the following commands:

```bash
python -m recsys.cli download   # Downloads the MovieLens latest-small dataset (~1 MB)
python -m recsys.cli prepare    # Filters users, splits train/test, persists artefacts
```

### 3. Train and evaluate the model

Train the recommender model. The model training process also includes evaluation on key metrics like precision, recall, and AUC.

```bash
python -m recsys.cli train
```

After training, the CLI prints key evaluation metrics (precision@k, recall@k, and AUC)
so you can track performance and iterate on hyper-parameters. You can re-run the
command with options such as `--loss bpr` or `--num-components 128` to explore
alternatives.

Use `python -m recsys.cli --help` at any time to inspect available commands and options.

### 4. Generate recommendations

Generate personalized recommendations for a user based on the trained model.

```bash
python -m recsys.cli recommend 1 --k 5
```

Replace `1` with any user ID present in the dataset. You can also export the recommendations to a JSON file for further use or integration.

```bash
python -m recsys.cli recommend 1 --k 5 --export output.json
```

## Project structure

Here’s an overview of the project structure:

```
.
├── requirements.txt          # Python dependencies (LightFM, pandas, Typer, ...)
├── src/
│   └── recsys/
│       ├── cli.py            # Typer-based command line interface
│       ├── config.py         # Shared paths and configuration constants
│       ├── data.py           # Downloading, filtering, and interaction matrix builders
│       ├── evaluation.py     # Convenience wrapper around evaluation metrics
│       └── model.py          # Training loop, evaluation, persistence, recommendations
├── data/                     # Generated artefacts (ignored by Git)
├── models/                   # Persisted LightFM weights (ignored by Git)
└── notebooks/                # Optional experiments / notebooks (ignored by Git)
```

## Customising the pipeline

You can customize various parts of the pipeline to fit your needs:

- **Minimum activity thresholds** – use `python -m recsys.cli prepare --min-ratings-per-user 10`
  to focus on power users.
- **Loss functions** – switch between `warp`, `bpr`, or `logistic` losses via
  `python -m recsys.cli train --loss bpr`.
- **Latent dimensionality** – tune `--num-components` to balance runtime and model
  capacity.
- **Recommendation exports** – supply `--export` to the `recommend` command to create
  shareable JSON files.

## Next steps

To take this project further, consider:

- Adding offline evaluation notebooks or dashboards (e.g., with Jupyter or Streamlit).
- Tracking experiments with a tool such as MLflow or Weights & Biases.
- Serving the trained model through an API (FastAPI) and integrating it into a front-end.
- Experimenting with sequence-aware or contextual recommenders for richer signals.

```

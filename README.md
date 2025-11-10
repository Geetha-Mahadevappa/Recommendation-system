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
- **Dataset insights** – Quickly inspect dataset health with built-in descriptive statistics.
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
`````

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

Once the data has been prepared you can inspect dataset statistics:

```bash
python -m recsys.cli describe
```

Add `--as-json` to the command above to export the summary in machine-readable form.

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
```

---

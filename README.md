# FRAME — Hybrid Movie Recommendation System

A movie discovery application with a React frontend and FastAPI backend, powered by a hybrid content-based recommendation system.

## Features

- Movie recommendations based on a selected movie
- Semantic similarity using BGE sentence embeddings
- Eight TF-IDF feature representations
- Configurable feature weights
- Filter-based movie discovery
- Interactive React interface

## Tech Stack

**Frontend:** React, Vite, JavaScript, CSS

**Backend:** Python, FastAPI, Uvicorn

**Recommendation Model:** Sentence Transformers, BGE, TF-IDF, scikit-learn

**Data Processing:** pandas, NumPy

## Recommendation Approach

The recommender combines semantic similarity from movie overviews with TF-IDF similarity across eight metadata features:

- Genres
- Keywords
- Cast
- Director
- Writers
- Production
- Country
- Language

The final ranking uses a weighted combination of these similarity signals.

## Project Structure

- `frontend/` — React application
- `backend/` — FastAPI endpoints and recommendation logic
- `notebooks/` — Model experiments
- `Data/` — Local movie dataset; excluded from Git
- `tests/` — Automated tests
- `docs/` — Technical documentation

## Setup

### Prerequisites

- Python 3.12
- Node.js LTS and npm
- The required TMDB-derived movie CSV dataset

### Backend

From the project root, create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

Set the dataset path in PowerShell:

```powershell
$env:TMDB_DATA_PATH = (Resolve-Path "Data/tmdb/tmdb_movies_Dataset.csv").Path
```

Start the API:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

### Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, typically http://localhost:5173.

## Dataset

The dataset is not included in this repository. Place the required CSV at:

`Data/tmdb/tmdb_movies_Dataset.csv`

Alternatively, set `TMDB_DATA_PATH` to its actual location.

## Notes

- The first model initialization may take time and memory.
- The BGE model may need to be downloaded on first use.
- Keep local datasets, virtual environments, model caches, and secrets out of version control.
- Verify the recommender's results against the original implementation when changing its code.

## License

Add your chosen license before distributing the project.

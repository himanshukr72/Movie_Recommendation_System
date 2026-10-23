# FRAME — Hybrid Movie Recommendation System

A movie discovery application built with **React, FastAPI, and Python**, combining semantic similarity and metadata-based recommendations to help users discover movies with similar stories, genres, cast, and production characteristics.

---

## Overview

FRAME is a hybrid content-based movie recommendation system that combines **BGE sentence embeddings** with eight TF-IDF feature representations. It uses movie plot descriptions and metadata to identify movies with similar characteristics.

The application provides an interactive React frontend backed by a FastAPI service, allowing users to select a movie, refine their discovery preferences, and explore recommendations.

The project uses a merged TMDB-derived movie metadata and credits dataset as its primary data source.

## Features

- **Hybrid recommendations:** Combines semantic similarity and metadata-based similarity.
- **Semantic search:** Uses BGE sentence embeddings to represent movie overviews.
- **Multi-feature similarity:** Uses eight TF-IDF representations of movie metadata.
- **Customizable feature weights:** Supports adjusting the relative contribution of recommendation features.
- **Filter-based discovery:** Supports exploration by directors, genres, cast, writers, keywords, production companies, countries, languages, and plot descriptions.
- **Interactive interface:** React-based frontend with movie selection and recommendation results.
- **REST API:** FastAPI endpoints for health checks, movie options, and recommendation requests.

## Tech Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Semantic embeddings | Sentence Transformers, BGE |
| Text vectorization | TF-IDF, scikit-learn |
| Data processing | pandas, NumPy |
| Dataset | TMDB-derived movie metadata and credits |
| Development tools | Git, GitHub, Jupyter Notebook |

## Recommendation Approach

FRAME combines semantic similarity from movie overviews with TF-IDF-based similarity across eight metadata features.

### 1. Semantic similarity

The system uses the `BAAI/bge-small-en-v1.5` sentence-transformer model to generate embeddings from movie overview text. These embeddings capture semantic relationships between movie descriptions beyond exact keyword matches.

### 2. Metadata-based similarity

Separate TF-IDF representations are created for the following features:

| Feature | Default weight |
|---|---:|
| Genres | 0.18 |
| Keywords | 0.13 |
| Cast | 0.10 |
| Director | 0.06 |
| Writers | 0.04 |
| Production | 0.04 |
| Production country | 0.03 |
| Original language | 0.02 |

The semantic overview similarity has a default weight of **0.40**.

The weighted similarity signals are combined to rank movies according to their similarity to the selected movie or discovery preferences.

These are the configured default weights; recommendation quality should be evaluated against the intended use case and validated through testing.

## Dataset: TMDB Movie Metadata and Credits

### Dataset overview

The dataset contains movie metadata and credits information collected through the TMDB API. Movie metadata and credits tables were merged using their movie identifiers to create a unified CSV suitable for data analysis, natural language processing, and recommendation experiments.

The merged dataset is not an official TMDB publication.

### Dataset contents

The dataset may include the following information:

- **Movie metadata:** Movie ID, title, original title, original language, release date, runtime, and status.
- **Textual information:** Plot overview, tagline, and homepage.
- **Genres and keywords:** Movie genres and descriptive tags.
- **Cast and crew:** Actors, directors, writers, and other production roles.
- **Production details:** Production companies, production countries, and spoken languages.
- **Financial information:** Budget and revenue, where available.
- **Audience and popularity metrics:** Vote average, vote count, and popularity.

Some fields—including genres, keywords, cast, crew, production companies, production countries, and spoken languages—may contain JSON-formatted strings and require parsing before analysis.

### Dataset preparation

The merged dataset is intended to provide a unified starting point for recommendation-system development and exploratory analysis.

Before using the dataset, consider checking:

- Missing values and incomplete movie descriptions
- Duplicate movie identifiers
- Data types and inconsistent formatting
- Nested JSON fields that require parsing
- Availability of metadata across different movies

### Dataset location

The application expects the CSV file at:

```text
Data/
└── tmdb/
    └── tmdb_movies_Dataset.csv
```

The dataset is excluded from this repository because of its size. You must obtain the dataset separately and place it at the expected path, or configure a custom path through the `TMDB_DATA_PATH` environment variable.

### Potential use cases

The dataset can support:

1. Content-based movie recommendation
2. Natural language processing of movie descriptions
3. Exploratory data analysis of genres, release dates, popularity, and ratings
4. Similarity analysis using cosine similarity and other distance measures
5. Feature engineering for hybrid recommendation systems
6. Interactive dashboards for exploring movie characteristics

### Dataset limitations

Data availability and completeness may vary across movies. Financial information, descriptions, credits, and other fields may be missing or incomplete.

Popularity and voting metrics can change over time, so the dataset reflects the information available when it was collected.

### Data source and attribution

The data was collected through the TMDB API. Before redistributing the dataset or using it in a public or commercial application, consult TMDB's applicable API terms, attribution requirements, and data-use policies.

This dataset is not an official TMDB publication. Use of TMDB-derived data is subject to the applicable permissions and terms of the source.

## Project Structure

```text
Movie_Recommendation_System/
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── .env.example
│
├── Data/
│   └── tmdb/
│       └── .gitkeep
│
├── notebooks/
│   └── Movie_Recommendation.ipynb
│
├── tests/
├── docs/
├── .gitignore
├── README.md
└── LICENSE
```

*The tree above describes the intended organization. Optional folders such as `tests/`, `docs/`, and `notebooks/` should be included only if they exist or you create them.*

## Getting Started

### Prerequisites

Install the following before running the application:

- Python 3.12
- Node.js LTS and npm
- Git
- The required TMDB-derived movie CSV dataset

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Movie_Recommendation_System
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Prepare the dataset

Place the dataset at:

```text
Data/tmdb/tmdb_movies_Dataset.csv
```

Alternatively, set `TMDB_DATA_PATH` to the full path of the CSV file.

### 3. Set up the backend

Create a virtual environment from the project root:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

If activation succeeds, install the dependencies with:

```bash
python -m pip install -r backend/requirements.txt
```

Set the dataset path in PowerShell if necessary:

```powershell
$env:TMDB_DATA_PATH = (Resolve-Path "Data/tmdb/tmdb_movies_Dataset.csv").Path
```

Start the API from the project root:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If you did not activate the virtual environment, use:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation should be available at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Set up the frontend

Open a second terminal and navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

If PowerShell blocks `npm.ps1`, use the Windows command wrapper:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite, typically:

[http://localhost:5173](http://localhost:5173)

Keep both the backend and frontend running while using the application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check backend health and dataset availability |
| GET | `/api/options` | Retrieve movie and filter options |
| POST | `/api/recommend` | Generate movie recommendations |

Interactive endpoint testing is available through the FastAPI documentation at `/docs`.

## Performance Considerations

- The first model initialization may take time because the dataset must be loaded and the recommendation features prepared.
- Downloading the BGE model may be necessary on first use.
- Dataset processing, TF-IDF matrices, and embeddings can require substantial memory.
- Production deployments should consider persistent model artifacts, caching, startup behavior, and available memory.
- Validate recommendations against the original implementation when modifying the recommendation logic.

## Git and Data Management

The following should generally remain outside version control:

- Python virtual environments such as `.venv/`
- Frontend dependencies such as `frontend/node_modules/`
- Large local dataset files
- Generated model artifacts and caches
- Environment files containing secrets or credentials

Commit dependency manifests, source code, documentation, and appropriate configuration examples instead.

## License

Add a `LICENSE` file specifying the license you intend to use for your own code. A license for the application does not automatically grant redistribution rights for the underlying TMDB-derived dataset or its contents.

## Acknowledgements

- [The Movie Database (TMDB)](https://www.themoviedb.org/) — movie metadata and credits data source
- [Sentence Transformers](https://www.sbert.net/) — sentence embedding framework
- [Hugging Face](https://huggingface.co/) — model hosting and ecosystem
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [React](https://react.dev/) and [Vite](https://vite.dev/) — frontend technologies


## Live Demo

- **Frontend (React + Vite):** http://localhost:5173/
- **Backend API (FastAPI):** http://127.0.0.1:8000/docs?utm_source=chatgpt.com#/default/options_api_options_get



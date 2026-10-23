from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from tempfile import gettempdir

import gdown
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Streamlit setup and paths
# ============================================================
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
GOOGLE_DRIVE_FILE_ID = "1dpy1mRNb9TtrTTsB6tB-KwJbEzINwsvr"
DATA_PATH = Path(gettempdir()) / "tmdb_movies_Dataset.csv"
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

DEFAULT_WEIGHTS = {
    "bge": 0.40,
    "genres": 0.18,
    "keywords": 0.13,
    "cast": 0.10,
    "director": 0.06,
    "writers": 0.04,
    "production": 0.04,
    "country": 0.03,
    "language": 0.02,
}



@st.cache_resource(show_spinner="Downloading movie dataset from Google Drive...")
def ensure_dataset() -> str:
    """Download the combined CSV to temporary storage when it is not present."""
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        return str(DATA_PATH)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = gdown.download(
        id=GOOGLE_DRIVE_FILE_ID,
        output=str(DATA_PATH),
        quiet=False,
    )

    if not downloaded_path or not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        raise RuntimeError(
            "Google Drive did not provide the CSV. Confirm the file is shared as "
            "'Anyone with the link — Viewer' and that download limits have not been reached."
        )

    return str(DATA_PATH)


# ============================================================
# Notebook preprocessing helpers (based on cells 12, 14, 16, 17)
# ============================================================
def parse_tmdb_list(value):
    """Convert a JSON string into a Python list safely."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def get_names(value, limit=None):
    """Extract unique names from a list of TMDB dictionaries."""
    names, seen = [], set()
    for item in parse_tmdb_list(value):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            name = name.strip()
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names if limit is None else names[:limit]


def get_director(value):
    for member in parse_tmdb_list(value):
        if isinstance(member, dict) and member.get("job") == "Director":
            name = member.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""


def get_writers(value):
    writer_jobs = {"Writer", "Screenplay", "Story"}
    writers, seen = [], set()
    for member in parse_tmdb_list(value):
        if not isinstance(member, dict) or member.get("job") not in writer_jobs:
            continue
        name = member.get("name")
        if isinstance(name, str) and name.strip():
            name = name.strip()
            if name not in seen:
                writers.append(name)
                seen.add(name)
    return writers


def clean_token(value):
    """Normalize a metadata value into lowercase text, matching the notebook."""
    if value is None:
        return ""
    if not isinstance(value, str):
        if pd.isna(value):
            return ""
        value = str(value)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def as_clean_text_list(values):
    if not isinstance(values, list):
        return ""
    return " ".join(clean_token(v) for v in values if str(v).strip())


# ============================================================
# Build the same feature tables and model from the combined CSV
# ============================================================
@st.cache_resource(show_spinner=False)
def load_recommender(csv_path: str):
    movies = pd.read_csv(csv_path, low_memory=False)

    # This file is already merged, so do NOT merge tmdb_movies and tmdb_credits again.
    if "movie_id" not in movies.columns:
        if "id" in movies.columns:
            movies["movie_id"] = movies["id"]
        else:
            raise ValueError("Dataset must contain either 'movie_id' or 'id'.")
    if "title" not in movies.columns:
        raise ValueError("Dataset must contain a 'title' column.")
    if "crew" not in movies.columns:
        movies["crew"] = "[]"
    if "cast" not in movies.columns:
        movies["cast"] = "[]"

    movies = movies.dropna(subset=["movie_id", "title"]).copy()
    movies["title"] = movies["title"].astype(str).str.strip()
    movies = movies[movies["title"].ne("")].drop_duplicates(subset=["movie_id"], keep="first").copy()
    movies["overview"] = movies.get("overview", pd.Series("", index=movies.index)).fillna("").astype(str)

    list_columns = {
        "genres": "genres_list",
        "keywords": "keywords_list",
        "production_companies": "production_companies_list",
        "production_countries": "production_countries_list",
    }
    for source_col, output_col in list_columns.items():
        if source_col in movies.columns:
            movies[output_col] = movies[source_col].apply(get_names)
        else:
            movies[output_col] = [[] for _ in range(len(movies))]

    movies["cast_top5"] = movies["cast"].apply(lambda value: get_names(value, limit=5))
    movies["director"] = movies["crew"].apply(get_director)
    movies["writers"] = movies["crew"].apply(get_writers)
    movies["release_year"] = pd.to_datetime(
        movies.get("release_date", pd.Series("", index=movies.index)), errors="coerce"
    ).dt.year

    for column in ["runtime", "vote_average", "vote_count", "popularity", "budget", "revenue"]:
        if column in movies.columns:
            movies[column] = pd.to_numeric(movies[column], errors="coerce")
        else:
            movies[column] = np.nan

    movies["original_language"] = movies.get(
        "original_language", pd.Series("", index=movies.index)
    ).fillna("").astype(str)
    movies["poster_path"] = movies.get(
        "poster_path", pd.Series("", index=movies.index)
    ).fillna("").astype(str)

    model_columns = [
        "movie_id", "title", "overview", "genres_list", "keywords_list",
        "cast_top5", "director", "writers", "production_companies_list",
        "production_countries_list", "original_language", "release_year",
        "runtime", "vote_average", "vote_count", "popularity", "budget",
        "revenue", "poster_path",
    ]
    model_df = movies[[c for c in model_columns if c in movies.columns]].copy().reset_index(drop=True)

    feature_text = pd.DataFrame(index=model_df.index)
    feature_text["overview"] = model_df["overview"].fillna("").astype(str)
    feature_text["genres"] = model_df["genres_list"].apply(as_clean_text_list)
    feature_text["keywords"] = model_df["keywords_list"].apply(as_clean_text_list)
    feature_text["cast"] = model_df["cast_top5"].apply(as_clean_text_list)
    feature_text["director"] = model_df["director"].apply(clean_token)
    feature_text["writers"] = model_df["writers"].apply(as_clean_text_list)
    feature_text["production"] = model_df["production_companies_list"].apply(as_clean_text_list)
    feature_text["country"] = model_df["production_countries_list"].apply(as_clean_text_list)
    feature_text["language"] = model_df["original_language"].fillna("").astype(str)

    # Structured TF-IDF matrices, matching notebook cell 22.
    feature_vectorizers, feature_matrices = {}, {}
    structured_features = [
        "genres", "keywords", "cast", "director",
        "writers", "production", "country", "language",
    ]
    for feature in structured_features:
        vectorizer = TfidfVectorizer(
            tokenizer=str.split, preprocessor=None, token_pattern=None,
            lowercase=True, norm="l2"
        )
        matrix = vectorizer.fit_transform(feature_text[feature].fillna(""))
        feature_vectorizers[feature] = vectorizer
        feature_matrices[feature] = matrix

    # BGE overview embeddings, matching notebook cell 21.
    bge_model = SentenceTransformer(BGE_MODEL_NAME)
    bge_inputs = feature_text["overview"].replace("", "No overview available").tolist()
    bge_embeddings = bge_model.encode(
        bge_inputs, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    return movies, model_df, feature_text, feature_vectorizers, feature_matrices, bge_model, bge_embeddings


def normalize_weights(weights):
    clean_weights = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(clean_weights.values())
    if total <= 0:
        raise ValueError("At least one feature weight must be positive.")
    return {key: value / total for key, value in clean_weights.items()}


def find_movie_index(model_df, movie_title):
    matches = model_df.index[
        model_df["title"].astype(str).str.casefold() == str(movie_title).casefold()
    ].tolist()
    if not matches:
        raise ValueError(f"Movie title not found: {movie_title}")
    return matches[0]


def recommend_movies(
    model_df, feature_text, feature_vectorizers, feature_matrices,
    bge_model, bge_embeddings, movie_title=None, director=None, genres=None,
    writers=None, production=None, cast=None, keywords=None, country=None,
    language=None, overview=None, n=5, weights=None,
):
    """Notebook's hybrid recommendation logic, exposed to Streamlit."""
    if n < 1:
        raise ValueError("n must be at least 1.")

    criteria = {
        "director": director, "genres": genres, "writers": writers,
        "production": production, "cast": cast, "keywords": keywords,
        "country": country, "language": language,
    }

    def to_query_text(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        text = str(value).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    seed_index = find_movie_index(model_df, movie_title) if movie_title else None
    query_features = {}
    if seed_index is not None:
        for feature in feature_matrices:
            query_features[feature] = to_query_text(feature_text.iloc[seed_index][feature])

    for feature, value in criteria.items():
        if value is not None:
            query_features[feature] = to_query_text(value)

    weight_config = DEFAULT_WEIGHTS.copy()
    if weights is not None:
        weight_config.update(weights)
    weight_config = {key: max(0.0, float(value)) for key, value in weight_config.items()}

    combined_scores = np.zeros(len(model_df), dtype=np.float32)
    total_weight = 0.0

    for feature, query_text in query_features.items():
        if not query_text or feature not in feature_matrices:
            continue
        weight = weight_config.get(feature, 0.0)
        if weight <= 0:
            continue

        query_vector = feature_vectorizers[feature].transform([query_text])
        if query_vector.nnz == 0:
            # Preserve the notebook's director fallback, with a tolerant contains match.
            if feature == "director" and criteria.get("director"):
                director_query = str(criteria["director"]).casefold().strip()
                director_mask = model_df["director"].fillna("").str.casefold().str.contains(
                    re.escape(director_query), na=False
                )
                if director_mask.any():
                    combined_scores += weight * director_mask.to_numpy(dtype=np.float32)
                    total_weight += weight
            continue

        scores = cosine_similarity(query_vector, feature_matrices[feature]).ravel()
        combined_scores += weight * scores.astype(np.float32)
        total_weight += weight

    if overview is not None and str(overview).strip():
        query_embedding = bge_model.encode(
            [str(overview).strip()], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        use_bge = True
    elif seed_index is not None:
        query_embedding = bge_embeddings[seed_index]
        use_bge = True
    else:
        use_bge = False

    bge_weight = weight_config.get("bge", 0.0)
    if use_bge and bge_weight > 0:
        bge_scores = bge_embeddings @ query_embedding
        combined_scores += bge_weight * bge_scores.astype(np.float32)
        total_weight += bge_weight

    if total_weight <= 0:
        raise ValueError(
            "No usable query features found. Choose a movie or provide a valid search criterion."
        )

    combined_scores /= total_weight
    if seed_index is not None:
        combined_scores[seed_index] = -np.inf

    indices = np.argsort(combined_scores)[::-1][:n]
    result = model_df.iloc[indices][
        ["movie_id", "title", "genres_list", "release_year", "overview",
         "vote_average", "vote_count", "popularity", "poster_path",
         "director", "runtime"]
    ].copy()
    result["similarity_score"] = combined_scores[indices]
    return result.reset_index(drop=True)


# ============================================================
# Streamlit interface
# ============================================================
st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Your notebook's BGE + feature-specific TF-IDF hybrid recommender, wrapped in Streamlit.")

try:
    dataset_path = ensure_dataset()
except Exception as exc:
    st.error(f"Could not obtain the movie dataset: {exc}")
    st.stop()

with st.spinner("Loading dataset, TF-IDF feature matrices, and BGE embeddings. The first run can take a while..."):
    try:
        movies, model_df, feature_text, feature_vectorizers, feature_matrices, bge_model, bge_embeddings = load_recommender(dataset_path)
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.success(f"Loaded {len(model_df):,} movies and built the hybrid recommendation model.")
st.caption(f"Dataset: Google Drive CSV · BGE model: `{BGE_MODEL_NAME}`")

titles = sorted(model_df["title"].dropna().astype(str).unique().tolist())
directors = sorted({x for x in model_df["director"].dropna().astype(str) if x.strip()})
genres_all = sorted({x for values in model_df["genres_list"] for x in values})
countries_all = sorted({x for values in model_df["production_countries_list"] for x in values})
languages_all = sorted({x for x in model_df["original_language"].dropna().astype(str) if x.strip()})

with st.sidebar:
    st.header("Find recommendations")
    movie_title = st.selectbox("Starting movie (optional)", [""] + titles)
    st.caption("Choose a movie for similarity recommendations, or search using filters below.")
    director = st.selectbox("Director", [""] + directors)
    genres = st.multiselect("Genres", genres_all)
    cast = st.text_input("Cast member", placeholder="e.g. Tom Holland")
    writers = st.text_input("Writer", placeholder="e.g. Christopher Nolan")
    production = st.text_input("Production company")
    keywords = st.text_input("Keywords", placeholder="e.g. superhero, space")
    country = st.selectbox("Production country", [""] + countries_all)
    language = st.selectbox("Original language", [""] + languages_all)
    overview = st.text_area("Plot / overview query (optional)", height=100)
    n = st.slider("Number of recommendations", 5, 30, 10)

    with st.expander("Hybrid model weights", expanded=False):
        st.write("Defaults match the notebook's `DEFAULT_WEIGHTS`.")
        weights = {}
        for feature, default in DEFAULT_WEIGHTS.items():
            weights[feature] = st.slider(
                feature.upper() if feature == "bge" else feature.title(),
                min_value=0.0, max_value=1.0, value=float(default), step=0.01,
                key=f"weight_{feature}",
            )

run = st.button("Recommend movies", type="primary", use_container_width=True)

if run:
    if not movie_title and not any([director, genres, cast.strip(), writers.strip(),
                                    production.strip(), keywords.strip(), country,
                                    language, overview.strip()]):
        st.warning("Choose a starting movie or provide at least one search criterion.")
    else:
        try:
            with st.spinner("Scoring movies with the hybrid model..."):
                results = recommend_movies(
                    model_df=model_df,
                    feature_text=feature_text,
                    feature_vectorizers=feature_vectorizers,
                    feature_matrices=feature_matrices,
                    bge_model=bge_model,
                    bge_embeddings=bge_embeddings,
                    movie_title=movie_title or None,
                    director=director or None,
                    genres=genres or None,
                    writers=writers.strip() or None,
                    production=production.strip() or None,
                    cast=cast.strip() or None,
                    keywords=keywords.strip() or None,
                    country=country or None,
                    language=language or None,
                    overview=overview.strip() or None,
                    n=n,
                    weights=weights,
                )
            st.subheader("Recommended movies")
            st.write(f"Showing {len(results)} recommendations.")
            cols = st.columns(5)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    with st.container(border=True):
                        poster = str(row.get("poster_path", "") or "").strip()
                        if poster and poster.lower() not in {"nan", "none"} and poster.startswith("/"):
                            st.image(f"{TMDB_IMAGE_BASE}{poster}", use_container_width=True)
                        else:
                            st.markdown("🎞️ *Poster unavailable*")
                        st.markdown(f"**{row['title']}**")
                        year = row.get("release_year")
                        year_text = str(int(year)) if pd.notna(year) else "Year unknown"
                        st.caption(f"{year_text} · Similarity {row['similarity_score']:.3f}")
                        genre_text = ", ".join(row["genres_list"]) if isinstance(row["genres_list"], list) else ""
                        if genre_text:
                            st.caption(genre_text)
                        rating = row.get("vote_average")
                        if pd.notna(rating):
                            st.write(f"⭐ {rating:.1f}/10")
                        director_text = str(row.get("director", "") or "").strip()
                        if director_text:
                            st.caption(f"Director: {director_text}")
                        runtime = row.get("runtime")
                        if pd.notna(runtime) and runtime > 0:
                            st.caption(f"{int(runtime)} min")
                        plot = str(row.get("overview", "") or "").strip()
                        if plot:
                            with st.expander("Plot"):
                                st.write(plot)

            with st.expander("View results as a table"):
                table = results.copy()
                table["genres"] = table["genres_list"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
                st.dataframe(
                    table[["title", "release_year", "genres", "director", "vote_average", "similarity_score"]],
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
else:
    st.info("Choose a movie or enter criteria in the sidebar, then click **Recommend movies**.")

with st.expander("About the model"):
    st.markdown(
        """
        This app preserves the notebook's hybrid approach:
        - **BGE embeddings** for semantic similarity between overviews.
        - **Separate TF-IDF matrices** for genres, keywords, cast, director, writers,
          production companies, countries, and language.
        - **Weighted score aggregation** using the notebook's default weights.

        Similarity scores are relative model scores, not predicted audience ratings.
        """
    )

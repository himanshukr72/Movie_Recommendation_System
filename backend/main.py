import os, json, re, unicodedata
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-small-en-v1.5"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
DATA_PATH = Path(os.getenv("TMDB_DATA_PATH", "Data/tmdb/tmdb_movies_Dataset.csv")).resolve()
DEFAULT_WEIGHTS = {"bge":.40,"genres":.18,"keywords":.13,"cast":.10,"director":.06,"writers":.04,"production":.04,"country":.03,"language":.02}
FEATURES = ["genres","keywords","cast","director","writers","production","country","language"]
app = FastAPI(title="TMDB Hybrid Movie Recommender API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS","http://localhost:5173").split(","), allow_methods=["*"], allow_headers=["*"])

def parse_list(v):
    if isinstance(v,list): return v
    if not isinstance(v,str) or not v.strip(): return []
    try:
        x=json.loads(v); return x if isinstance(x,list) else []
    except Exception: return []
def names(v,limit=None):
    out=[]; seen=set()
    for x in parse_list(v):
        n=x.get("name") if isinstance(x,dict) else None
        if isinstance(n,str) and n.strip() and n.strip() not in seen: out.append(n.strip()); seen.add(n.strip())
    return out if limit is None else out[:limit]
def director(v):
    for x in parse_list(v):
        if isinstance(x,dict) and x.get("job")=="Director" and x.get("name"): return x["name"].strip()
    return ""
def writers(v):
    out=[]; seen=set()
    for x in parse_list(v):
        if isinstance(x,dict) and x.get("job") in {"Writer","Screenplay","Story"} and x.get("name") and x["name"].strip() not in seen:
            out.append(x["name"].strip()); seen.add(x["name"].strip())
    return out
def clean(v):
    if v is None: return ""
    if not isinstance(v,str):
        if pd.isna(v): return ""
        v=str(v)
    v=unicodedata.normalize("NFKD",v).encode("ascii","ignore").decode().lower().strip()
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",v)).strip()
def clean_list(v): return " ".join(clean(x) for x in v) if isinstance(v,list) else ""

@lru_cache(maxsize=1)
def load_model():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Set TMDB_DATA_PATH.")
    movies=pd.read_csv(DATA_PATH,low_memory=False)
    if "movie_id" not in movies and "id" in movies: movies["movie_id"]=movies["id"]
    if "movie_id" not in movies or "title" not in movies: raise ValueError("CSV needs movie_id (or id) and title columns.")
    for c in ["cast","crew"]:
        if c not in movies: movies[c]="[]"
    movies=movies.dropna(subset=["movie_id","title"]).copy()
    movies["title"]=movies["title"].astype(str).str.strip()
    movies=movies[movies.title.ne("")].drop_duplicates("movie_id").copy()
    movies["overview"]=movies.get("overview",pd.Series("",index=movies.index)).fillna("").astype(str)
    mapping={"genres":"genres_list","keywords":"keywords_list","production_companies":"production_companies_list","production_countries":"production_countries_list"}
    for src,dst in mapping.items(): movies[dst]=movies[src].apply(names) if src in movies else [[] for _ in range(len(movies))]
    movies["cast_top5"]=movies["cast"].apply(lambda x:names(x,5)); movies["director"]=movies["crew"].apply(director); movies["writers"]=movies["crew"].apply(writers)
    movies["release_year"]=pd.to_datetime(movies.get("release_date",pd.Series("",index=movies.index)),errors="coerce").dt.year
    for c in ["runtime","vote_average","vote_count","popularity"]:
        movies[c]=pd.to_numeric(movies[c],errors="coerce") if c in movies else np.nan
    movies["original_language"]=movies.get("original_language",pd.Series("",index=movies.index)).fillna("").astype(str)
    movies["poster_path"]=movies.get("poster_path",pd.Series("",index=movies.index)).fillna("").astype(str)
    cols=["movie_id","title","overview","genres_list","keywords_list","cast_top5","director","writers","production_companies_list","production_countries_list","original_language","release_year","runtime","vote_average","vote_count","popularity","poster_path"]
    df=movies[cols].reset_index(drop=True).copy()
    ft=pd.DataFrame(index=df.index)
    ft["overview"]=df.overview
    ft["genres"]=df.genres_list.apply(clean_list); ft["keywords"]=df.keywords_list.apply(clean_list); ft["cast"]=df.cast_top5.apply(clean_list)
    ft["director"]=df.director.apply(clean); ft["writers"]=df.writers.apply(clean_list); ft["production"]=df.production_companies_list.apply(clean_list)
    ft["country"]=df.production_countries_list.apply(clean_list); ft["language"]=df.original_language
    vecs={}; mats={}
    for f in FEATURES:
        vecs[f]=TfidfVectorizer(tokenizer=str.split,preprocessor=None,token_pattern=None,lowercase=True,norm="l2")
        mats[f]=vecs[f].fit_transform(ft[f].fillna(""))
    bge=SentenceTransformer(MODEL_NAME)
    embeddings=bge.encode(ft.overview.replace("","No overview available").tolist(),batch_size=32,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True).astype("float32")
    return df,ft,vecs,mats,bge,embeddings

def recommend(req):
    df,ft,vecs,mats,bge,emb=load_model()
    seed=None
    if req.movie_title:
        found=df.index[df.title.str.casefold()==req.movie_title.casefold()].tolist()
        if not found: raise ValueError(f"Movie title not found: {req.movie_title}")
        seed=found[0]
    criteria={"director":req.director,"genres":req.genres,"writers":req.writers,"production":req.production,"cast":req.cast,"keywords":req.keywords,"country":req.country,"language":req.language}
    queries={}
    if seed is not None:
        for f in FEATURES: queries[f]=clean(ft.iloc[seed][f])
    for f,v in criteria.items():
        if v: queries[f]=clean(" ".join(v) if isinstance(v,list) else v)
    weights=DEFAULT_WEIGHTS.copy()
    if req.weights: weights.update(req.weights)
    scores=np.zeros(len(df),dtype=np.float32); total=0.
    for f,q in queries.items():
        w=max(0,float(weights.get(f,0)))
        if not q or w<=0: continue
        qv=vecs[f].transform([q])
        if qv.nnz:
            scores+=w*cosine_similarity(qv,mats[f]).ravel().astype(np.float32); total+=w
        elif f=="director" and req.director:
            mask=df.director.fillna("").str.casefold().str.contains(re.escape(req.director.casefold()),na=False)
            if mask.any(): scores+=w*mask.to_numpy(np.float32); total+=w
    use_bge=bool(req.overview and req.overview.strip()) or seed is not None
    if use_bge and weights.get("bge",0)>0:
        qemb=bge.encode([req.overview.strip()],convert_to_numpy=True,normalize_embeddings=True)[0] if req.overview and req.overview.strip() else emb[seed]
        scores+=max(0,float(weights["bge"]))*(emb@qemb).astype(np.float32); total+=max(0,float(weights["bge"]))
    if total<=0: raise ValueError("No usable query features. Choose a movie or provide a valid search criterion.")
    scores/=total
    if seed is not None: scores[seed]=-np.inf
    ids=np.argsort(scores)[::-1][:req.n]; out=[]
    for i in ids:
        r=df.iloc[int(i)]
        def safe(v):
            if v is None: return None
            if isinstance(v,(np.integer,)): return int(v)
            if isinstance(v,(np.floating,)): return None if np.isnan(v) else float(v)
            if isinstance(v,list): return v
            return v
        poster=str(r.poster_path or "")
        out.append({"movie_id":safe(r.movie_id),"title":r.title,"genres":r.genres_list or [],"release_year":safe(r.release_year),
          "overview":r.overview,"vote_average":safe(r.vote_average),"vote_count":safe(r.vote_count),"popularity":safe(r.popularity),
          "poster_url":IMAGE_BASE+poster if poster.startswith("/") else None,"director":r.director,"runtime":safe(r.runtime),"similarity_score":float(scores[i])})
    return out

class Request(BaseModel):
    movie_title:str|None=None; director:str|None=None; genres:list[str]=Field(default_factory=list)
    writers:str|None=None; production:str|None=None; cast:str|None=None; keywords:str|None=None
    country:str|None=None; language:str|None=None; overview:str|None=None
    n:int=Field(default=12,ge=1,le=50); weights:dict[str,float]|None=None

@app.get("/api/health")
def health():
    return {"status":"ok","dataset_exists":DATA_PATH.exists(),"dataset_path":str(DATA_PATH),"model":"BGE + eight TF-IDF matrices"}
@app.get("/api/options")
def options():
    try: df,*_=load_model()
    except Exception as e: raise HTTPException(503,detail=str(e))
    return {"titles":sorted(df.title.dropna().unique().tolist()),"directors":sorted({x for x in df.director if x}),
      "genres":sorted({x for xs in df.genres_list for x in xs}),"countries":sorted({x for xs in df.production_countries_list for x in xs}),
      "languages":sorted({x for x in df.original_language if x}),"movie_count":len(df),"default_weights":DEFAULT_WEIGHTS}
@app.post("/api/recommend")
def get_recommendations(req:Request):
    if not any([req.movie_title,req.director,req.genres,req.writers,req.production,req.cast,req.keywords,req.country,req.language,req.overview]):
        raise HTTPException(400,detail="Choose a movie or provide at least one search criterion.")
    try: return {"results":recommend(req)}
    except Exception as e: raise HTTPException(400,detail=str(e))

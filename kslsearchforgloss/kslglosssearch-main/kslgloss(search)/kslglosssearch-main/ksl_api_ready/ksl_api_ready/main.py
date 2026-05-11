from __future__ import annotations

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, validator
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime
import time
import asyncio
from functools import lru_cache
import os

if TYPE_CHECKING:
    from ksl_search_final import SignDatabaseSearcher

API_KEY = os.getenv("API_KEY", "super-secret-key")

app = FastAPI(
    title="KSL Sign Language API",
    description="Multi-strategy search for Kenya Sign Language (KSL) animations",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

try:
    from ksl_search_final import SignDatabaseSearcher
except ImportError:
    pass

searcher: Optional[SignDatabaseSearcher] = None


@app.on_event("startup")
async def startup():
    global searcher
    try:
        from ksl_search_final import SignDatabaseSearcher
        print("Loading KSL SignDatabaseSearcher...")
        searcher = SignDatabaseSearcher(
            db_file=os.getenv("DB_FILE", "./data/sign_database.json"),
            emb_file=os.getenv("EMB_FILE", "./data/embeddings.npy"),
            ids_file=os.getenv("IDS_FILE", "./data/sign_ids.json"),
            gloss_file=os.getenv("GLOSS_FILE", "./data/gloss_to_index.json"),
            spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm"),
        )
        print(f"Ready! {len(searcher._db)} signs loaded.")
        print(f"Semantic search: {'enabled' if searcher.is_embedding_model_ready() else 'disabled (exact/synonym only)'}")
    except ImportError:
        print("SignDatabaseSearcher not available")
        searcher = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    batch_semantic: bool = True
    semantic_threshold: float = 0.7

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("top_k")
    def top_k_range(cls, v):
        if v < 1 or v > 20:
            raise ValueError("top_k must be between 1 and 20")
        return v

    @validator("semantic_threshold")
    def threshold_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("semantic_threshold must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1000)
def _cached_search(query: str, top_k: int, batch_semantic: bool, semantic_threshold: float):
    return searcher.search(
        query,
        top_k=top_k,
        batch_semantic=batch_semantic,
        semantic_threshold=semantic_threshold,
    )


def _check_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Add header: X-API-Key: <your-key>",
        )


def _check_ready():
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search engine not ready yet.")


@app.post("/search")
async def search(req: SearchRequest, api_key: str = Security(api_key_header)):
    _check_api_key(api_key)
    _check_ready()

    start = time.time()
    loop = asyncio.get_event_loop()
    results, metadata = await loop.run_in_executor(
        None,
        lambda: _cached_search(
            req.query, req.top_k, req.batch_semantic, req.semantic_threshold
        ),
    )

    return {
        "query": req.query,
        "results": [r.to_dict() for r in results],
        "metadata": metadata,
        "total": len(results),
        "latency_ms": round((time.time() - start) * 1000, 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    ready = searcher is not None
    return {
        "status": "healthy" if ready else "loading",
        "semantic_search": searcher.is_embedding_model_ready() if ready else False,
        "signs_loaded": len(searcher._db) if ready else 0,
        "cache_stats": searcher.get_cache_stats() if ready else {},
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/cache/clear")
async def clear_cache(api_key: str = Security(api_key_header)):
    _check_api_key(api_key)
    _check_ready()
    searcher.clear_cache()
    _cached_search.cache_clear()
    return {"status": "cleared", "timestamp": datetime.now().isoformat()}


@app.get("/stats")
async def get_stats():
    _check_ready()
    return {
        "total_signs": len(searcher._db),
        "glosses_indexed": len(searcher._gloss_index),
        "synonyms_indexed": len(searcher._synonym_index),
        "semantic_search_ready": searcher.is_embedding_model_ready(),
        "cache": searcher.get_cache_stats(),
        "version": "3.0.0",
    }


@app.get("/")
async def root():
    return {
        "name": "KSL Sign Language API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "search": "POST /search",
    }
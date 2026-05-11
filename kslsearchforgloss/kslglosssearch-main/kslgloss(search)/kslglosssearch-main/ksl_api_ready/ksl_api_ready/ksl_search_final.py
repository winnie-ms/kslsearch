from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Represents a single sign-language search result."""

    sign_id: str
    gloss: str
    word: str
    bvh_file: str
    synonyms: List[str]
    semantic_domain: str
    complexity_score: float
    match_type: str                        # exact | synonym | semantic | proper_noun_letter
    similarity_score: Optional[float]
    query_gloss: str
    is_letter_from_proper_noun: bool = False
    parent_proper_noun: str = ""
    original_letter: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class SignDatabaseSearcher:
    """
    Search and retrieve sign-language animations from a JSON database.

    Search priority (per query token):
        1. Exact match   – gloss comparison (case-insensitive)
        2. Synonym match – word appears in a sign's synonym list
        3. Proper noun   – detected via spaCy NER → letter-by-letter fingerspelling
        4. Semantic      – FAISS inner-product similarity on normalised embeddings
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_file: str = "./data/sign_database.json",
        emb_file: str = "./data/embeddings.npy",
        ids_file: str = "./data/sign_ids.json",
        gloss_file: str = "./data/gloss_to_index.json",
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.db_file = db_file
        self.emb_file = emb_file
        self.ids_file = ids_file
        self.gloss_file = gloss_file
        self.spacy_model_name = spacy_model

        # Runtime state
        self._db: Dict[str, Any] = {}          # sign_id -> record
        self._gloss_index: Dict[str, str] = {} # lowercase gloss -> sign_id
        self._synonym_index: Dict[str, str] = {}  # lowercase synonym -> sign_id
        self._sign_ids: List[str] = []         # ordered list matching embeddings rows
        self._gloss_to_index: Dict[str, int] = {}  # gloss -> embedding row index

        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._embed_model = None
        self._nlp = None

        # Cache for proper-noun detection  {word -> bool}
        self._proper_noun_cache: Dict[str, bool] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_database()
        self._load_embeddings()
        self._load_spacy()
        self._load_embed_model()

    def _load_database(self) -> None:
        """Load sign_database.json and build gloss / synonym indexes."""
        if not os.path.exists(self.db_file):
            logger.warning("sign_database.json not found at %s", self.db_file)
            return

        with open(self.db_file, encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        for record in raw:
            sid = str(record.get("sign_id", record.get("id", "")))
            if not sid:
                continue
            self._db[sid] = record

            gloss = record.get("gloss", "").strip()
            if gloss:
                self._gloss_index[gloss.lower()] = sid

            for syn in record.get("synonyms", []):
                s = syn.strip().lower()
                if s and s not in self._synonym_index:
                    self._synonym_index[s] = sid

        logger.info("Loaded %d signs from database", len(self._db))

    def _load_embeddings(self) -> None:
        """Load pre-computed embeddings, sign_ids and gloss_to_index."""
        missing = [p for p in (self.emb_file, self.ids_file, self.gloss_file)
                   if not os.path.exists(p)]
        if missing:
            logger.warning("Embedding files not found: %s", missing)
            return

        self._embeddings = np.load(self.emb_file).astype("float32")

        # L2-normalise so inner-product == cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings /= norms

        with open(self.ids_file, encoding="utf-8") as f:
            self._sign_ids = json.load(f)

        with open(self.gloss_file, encoding="utf-8") as f:
            self._gloss_to_index = json.load(f)

        self._build_faiss_index()
        logger.info("Embeddings loaded: %s", self._embeddings.shape)

    def _build_faiss_index(self) -> None:
        try:
            import faiss  # type: ignore

            dim = self._embeddings.shape[1]
            n = self._embeddings.shape[0]

            if n > 10_000:
                nlist = min(int(n ** 0.5), 256)
                quantiser = faiss.IndexFlatIP(dim)
                idx = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                idx.train(self._embeddings)
            else:
                idx = faiss.IndexFlatIP(dim)

            idx.add(self._embeddings)
            self._faiss_index = idx
            logger.info("FAISS index built (%d vectors, dim=%d)", n, dim)
        except Exception as exc:
            logger.error("Could not build FAISS index: %s", exc)

    def _load_spacy(self) -> None:
        try:
            import spacy  # type: ignore
            self._nlp = spacy.load(self.spacy_model_name)
            logger.info("spaCy model '%s' loaded", self.spacy_model_name)
        except Exception as exc:
            logger.warning("spaCy unavailable (%s) – falling back to capitalisation heuristic", exc)

    def _load_embed_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name = os.getenv("EMBED_MODEL", "paraphrase-MiniLM-L3-v2")
            hf_token = os.getenv("HF_TOKEN")
            self._embed_model = SentenceTransformer(model_name, token=hf_token)
            logger.info("Embedding model '%s' loaded", model_name)
        except Exception as exc:
            logger.warning("Embedding model unavailable (%s) – semantic search disabled", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_embedding_model_ready(self) -> bool:
        """Return True when the sentence-transformer model is loaded."""
        return self._embed_model is not None and self._faiss_index is not None

    def clear_cache(self) -> None:
        """Clear the proper-noun detection cache."""
        self._proper_noun_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return statistics about the proper-noun cache."""
        proper = sum(1 for v in self._proper_noun_cache.values() if v)
        return {
            "cache_size": len(self._proper_noun_cache),
            "cached_proper_nouns": proper,
            "cached_common_words": len(self._proper_noun_cache) - proper,
        }

    # ------------------------------------------------------------------
    # Main search entry-point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 1,
        batch_semantic: bool = True,
        semantic_threshold: float = 0.7,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Search the database for all tokens in *query*.

        Returns
        -------
        results  : list of SearchResult (ordered by match-type priority)
        metadata : summary dict
        """
        tokens = query.strip().split()
        metadata: Dict[str, Any] = {
            "total_results": 0,
            "exact_matches": 0,
            "synonym_matches": 0,
            "semantic_matches": 0,
            "proper_noun_letter_matches": 0,
            "errors": [],
            "query_glosses": [t.lower() for t in tokens],
        }

        all_results: List[SearchResult] = []

        # Separate tokens that need semantic search
        semantic_needed: List[str] = []

        for token in tokens:
            try:
                result = self._search_token_exact_or_synonym(token, top_k)
                if result:
                    all_results.extend(result)
                    for r in result:
                        if r.match_type == "exact":
                            metadata["exact_matches"] += 1
                        else:
                            metadata["synonym_matches"] += 1
                    continue

                # Proper noun?
                if self._is_proper_noun(token):
                    letters = self._fingerspell(token)
                    all_results.extend(letters)
                    metadata["proper_noun_letter_matches"] += len(letters)
                    continue

                # Queue for semantic search
                semantic_needed.append(token)

            except Exception as exc:
                metadata["errors"].append(f"{token}: {exc}")

        # Batch or sequential semantic search
        if semantic_needed and self.is_embedding_model_ready():
            if batch_semantic:
                sem_results = self._batch_semantic_search(
                    semantic_needed, top_k, semantic_threshold
                )
            else:
                sem_results = []
                for tok in semantic_needed:
                    sem_results.extend(
                        self._semantic_search_one(tok, top_k, semantic_threshold)
                    )
            all_results.extend(sem_results)
            metadata["semantic_matches"] += len(sem_results)

        # Deduplicate by sign_id (keep first occurrence)
        seen: set = set()
        unique: List[SearchResult] = []
        for r in all_results:
            key = (r.sign_id, r.query_gloss)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Sort: exact > synonym > proper_noun_letter > semantic
        priority = {"exact": 0, "synonym": 1, "proper_noun_letter": 2, "semantic": 3}
        unique.sort(key=lambda r: priority.get(r.match_type, 99))

        metadata["total_results"] = len(unique)
        return unique, metadata

    # ------------------------------------------------------------------
    # Exact / synonym
    # ------------------------------------------------------------------

    def _search_token_exact_or_synonym(
        self, token: str, top_k: int
    ) -> List[SearchResult]:
        lower = token.lower()

        # 1. Exact match
        if lower in self._gloss_index:
            sid = self._gloss_index[lower]
            return [self._make_result(sid, lower, "exact", 1.0)]

        # 2. Synonym match
        if lower in self._synonym_index:
            sid = self._synonym_index[lower]
            return [self._make_result(sid, lower, "synonym", 0.95)]

        return []

    # ------------------------------------------------------------------
    # Proper noun / fingerspelling
    # ------------------------------------------------------------------

    def _is_proper_noun(self, word: str) -> bool:
        if word in self._proper_noun_cache:
            return self._proper_noun_cache[word]

        if self._nlp is not None:
            doc = self._nlp(word)
            result = any(ent.label_ in {"PERSON", "GPE", "ORG", "LOC", "NORP"}
                         for ent in doc.ents)
            if not result:
                # spaCy NER misses single capitalised tokens; use POS fallback
                result = any(tok.pos_ == "PROPN" for tok in doc)
        else:
            # Heuristic: starts with uppercase and is not sentence-start token
            result = word[0].isupper() and not word.isupper()

        self._proper_noun_cache[word] = result
        return result

    def _fingerspell(self, proper_noun: str) -> List[SearchResult]:
        results = []
        for ch in proper_noun.upper():
            if not ch.isalpha():
                continue
            sid = self._gloss_index.get(ch.lower(), f"letter_{ch}")
            r = self._make_result(
                sid, ch.lower(), "proper_noun_letter", 1.0,
                query_gloss=proper_noun.lower()
            )
            r.is_letter_from_proper_noun = True
            r.parent_proper_noun = proper_noun
            r.original_letter = ch
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def _batch_semantic_search(
        self,
        tokens: List[str],
        top_k: int,
        threshold: float,
    ) -> List[SearchResult]:
        try:
            vecs = self._embed_model.encode(tokens, convert_to_numpy=True).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs /= norms

            scores, indices = self._faiss_index.search(vecs, top_k)
            results = []
            for i, token in enumerate(tokens):
                for score, idx in zip(scores[i], indices[i]):
                    if idx < 0 or float(score) < threshold:
                        continue
                    sid = self._sign_ids[idx]
                    results.append(
                        self._make_result(sid, token, "semantic", float(score))
                    )
            return results
        except Exception as exc:
            logger.error("Batch semantic search failed: %s", exc)
            return []

    def _semantic_search_one(
        self, token: str, top_k: int, threshold: float
    ) -> List[SearchResult]:
        return self._batch_semantic_search([token], top_k, threshold)

    # ------------------------------------------------------------------
    # Helper: build SearchResult from sign_id
    # ------------------------------------------------------------------

    def _make_result(
        self,
        sign_id: str,
        query_gloss: str,
        match_type: str,
        similarity_score: Optional[float],
        *,
        query_gloss_override: Optional[str] = None,
    ) -> SearchResult:
        record = self._db.get(sign_id, {})
        return SearchResult(
            sign_id=sign_id,
            gloss=record.get("gloss", sign_id),
            word=record.get("word", query_gloss),
            bvh_file=record.get("bvh_file", ""),
            synonyms=record.get("synonyms", []),
            semantic_domain=record.get("semantic_domain", ""),
            complexity_score=float(record.get("complexity_score", 0.0)),
            match_type=match_type,
            similarity_score=similarity_score,
            query_gloss=query_gloss_override or query_gloss,
        )

    # Allow _make_result to accept query_gloss as positional-or-keyword
    # (Python alias so the fingerspell helper can pass it cleanly)
    def _make_result(  # noqa: F811  (intentional re-def for cleaner sig)
        self,
        sign_id: str,
        query_gloss: str,
        match_type: str,
        similarity_score: Optional[float],
        **kwargs,
    ) -> SearchResult:
        record = self._db.get(sign_id, {})
        r = SearchResult(
            sign_id=sign_id,
            gloss=record.get("gloss", sign_id),
            word=record.get("word", query_gloss),
            bvh_file=record.get("bvh_file", ""),
            synonyms=record.get("synonyms", []),
            semantic_domain=record.get("semantic_domain", ""),
            complexity_score=float(record.get("complexity_score", 0.0)),
            match_type=match_type,
            similarity_score=similarity_score,
            query_gloss=kwargs.get("query_gloss", query_gloss),
        )
        return r
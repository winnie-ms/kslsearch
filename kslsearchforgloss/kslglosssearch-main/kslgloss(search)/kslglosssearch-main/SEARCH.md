# KSL Search Engine Documentation

## Overview

`SignDatabaseSearcher` is a multi-strategy semantic search engine for Kenyan Sign Language (KSL) gloss retrieval.

**Key Components:**
- Exact gloss matching
- Synonym detection
- Proper noun fingerspelling (via spaCy NER)
- FAISS-based semantic similarity
- LRU caching for performance

---

## Installation

```bash
pip install sentence-transformers numpy faiss-cpu spacy
python -m spacy download en_core_web_sm
```

---

## Quick Start

```python
from ksl_search_final import SignDatabaseSearcher

# Initialize searcher
searcher = SignDatabaseSearcher(
    db_file="./data/sign_database.json",
    emb_file="./data/embeddings.npy",
    ids_file="./data/sign_ids.json",
    gloss_file="./data/gloss_to_index.json",
    spacy_model="en_core_web_sm"
)

# Search
results, metadata = searcher.search(
    query="Where is the bathroom",
    top_k=5,
    batch_semantic=True,
    semantic_threshold=0.7
)

# Display results
for result in results:
    print(f"{result.gloss} ({result.match_type}) - {result.similarity_score}")
```

---

## Search Strategy

### Priority Order (Per Token)

1. **Exact Match** (Highest Priority)
   - Case-insensitive gloss comparison
   - `similarity_score`: 1.0
   - Example: "HELLO" matches gloss "hello"

2. **Synonym Match**
   - Token found in sign's synonym list
   - `similarity_score`: 0.95
   - Example: "hi" matches sign with synonym "hi"

3. **Proper Noun Detection**
   - spaCy NER: PERSON, GPE, ORG, LOC, NORP
   - Fallback: Capitalization heuristic
   - Returns fingerspelling letters: J-O-H-N
   - `match_type`: "proper_noun_letter"

4. **Semantic Search** (Lowest Priority)
   - FAISS inner-product similarity on normalized embeddings
   - Configurable `semantic_threshold` (default: 0.7)
   - Batch processing for efficiency

---

## API

### SignDatabaseSearcher Class

#### Constructor

```python
SignDatabaseSearcher(
    db_file: str = "./data/sign_database.json",
    emb_file: str = "./data/embeddings.npy",
    ids_file: str = "./data/sign_ids.json",
    gloss_file: str = "./data/gloss_to_index.json",
    spacy_model: str = "en_core_web_sm"
)
```

**Parameters:**
- `db_file`: Path to JSON database with sign metadata
- `emb_file`: Pre-computed NumPy embeddings array
- `ids_file`: JSON mapping row indices to sign IDs
- `gloss_file`: JSON mapping glosses to embedding indices
- `spacy_model`: spaCy model for NER

#### Main Method: `search()`

```python
results, metadata = searcher.search(
    query: str,
    top_k: int = 1,
    batch_semantic: bool = True,
    semantic_threshold: float = 0.7
) -> Tuple[List[SearchResult], Dict[str, Any]]
```

**Parameters:**
- `query`: English sentence or phrase
- `top_k`: Number of results per token (1-20)
- `batch_semantic`: Use batch processing for semantic search
- `semantic_threshold`: Minimum similarity score (0.0-1.0)

**Returns:**
- `results`: List of `SearchResult` objects
- `metadata`: Summary statistics

#### SearchResult Dataclass

```python
@dataclass
class SearchResult:
    sign_id: str                      # Unique sign identifier
    gloss: str                        # KSL gloss notation
    word: str                         # Original word/query
    bvh_file: str                     # Path to animation file
    synonyms: List[str]               # Alternative terms
    semantic_domain: str              # Category (location, action, etc.)
    complexity_score: float           # Motion complexity (0.0-1.0)
    match_type: str                   # exact | synonym | semantic | proper_noun_letter
    similarity_score: Optional[float] # Match confidence (0.0-1.0)
    query_gloss: str                  # Original query token
    is_letter_from_proper_noun: bool  # Fingerspelling indicator
    parent_proper_noun: str           # Source proper noun (e.g., "John")
    original_letter: str              # Letter being fingerspelled
```

#### Utility Methods

```python
# Check if semantic search is ready
is_ready = searcher.is_embedding_model_ready() -> bool

# Clear proper noun cache
searcher.clear_cache() -> None

# Get cache statistics
stats = searcher.get_cache_stats() -> Dict[str, int]
```

---

## Data Format

### sign_database.json

```json
[
  {
    "sign_id": "001",
    "gloss": "BATHROOM",
    "word": "bathroom",
    "bvh_file": "animations/bathroom.bvh",
    "synonyms": ["restroom", "toilet", "wc"],
    "semantic_domain": "location",
    "complexity_score": 0.65
  },
  {
    "sign_id": "002",
    "gloss": "WHERE",
    "word": "where",
    "bvh_file": "animations/where.bvh",
    "synonyms": ["location", "place"],
    "semantic_domain": "interrogative",
    "complexity_score": 0.45
  }
]
```

### gloss_to_index.json

Maps gloss strings to embedding row indices:
```json
{
  "bathroom": 0,
  "where": 1,
  "hello": 2,
  ...
}
```

### sign_ids.json

Ordered list matching embedding rows:
```json
[
  "001",
  "002",
  "003",
  ...
]
```

### embeddings.npy

NumPy array (N x 384) of normalized embeddings using paraphrase-MiniLM-L3-v2.

---

## Output Examples

### Exact Match

```python
results, metadata = searcher.search("bathroom")

# Result:
SearchResult(
    sign_id="001",
    gloss="BATHROOM",
    word="bathroom",
    match_type="exact",
    similarity_score=1.0,
    query_gloss="bathroom"
)
```

### Synonym Match

```python
results, metadata = searcher.search("restroom")

# Result:
SearchResult(
    sign_id="001",           # Same sign, matched via synonym
    gloss="BATHROOM",
    word="bathroom",
    match_type="synonym",
    similarity_score=0.95,
    query_gloss="restroom"
)
```

### Proper Noun (Fingerspelling)

```python
results, metadata = searcher.search("John")

# Results: (list of letters)
SearchResult(
    sign_id="letter_J",
    gloss="J",
    match_type="proper_noun_letter",
    is_letter_from_proper_noun=True,
    parent_proper_noun="John",
    original_letter="J"
)
# ... J-O-H-N
```

### Semantic Match

```python
results, metadata = searcher.search("greetings", top_k=3, semantic_threshold=0.7)

# Result:
SearchResult(
    sign_id="002",
    gloss="HELLO",
    word="hello",
    match_type="semantic",
    similarity_score=0.82,
    query_gloss="greetings"
)
```

---

## Metadata Output

```json
{
  "total_results": 5,
  "exact_matches": 1,
  "synonym_matches": 0,
  "semantic_matches": 3,
  "proper_noun_letter_matches": 0,
  "errors": [],
  "query_glosses": ["bathroom", "where"]
}
```

---

## Caching

### Query Cache
- LRU cache: max 1,000 queries
- Automatic deduplication by (sign_id, query_gloss)
- Clear via: `searcher.clear_cache()`

### Proper Noun Cache
- Detects proper nouns (name, location, organization)
- Caches per session
- Statistics via: `searcher.get_cache_stats()`

**Example Stats:**
```python
{
    "cache_size": 150,
    "cached_proper_nouns": 45,
    "cached_common_words": 105
}
```

---

## Performance Tips

1. **Use Batch Semantic Search**
   ```python
   results, _ = searcher.search(query, batch_semantic=True)
   ```

2. **Increase top_k for recall**
   ```python
   results, _ = searcher.search(query, top_k=10)
   ```

3. **Adjust semantic_threshold**
   - `0.7`: Default (good balance)
   - `0.5`: Broader results, lower quality
   - `0.9`: Strict matching, fewer results

4. **Monitor cache statistics**
   ```python
   stats = searcher.get_cache_stats()
   if stats["cache_size"] > 800:
       searcher.clear_cache()
   ```

---

## Troubleshooting

### Embedding model not loading
```
⚠️ Embedding model unavailable – semantic search disabled
```
**Solution:** Install sentence-transformers
```bash
pip install sentence-transformers
```

### spaCy model missing
```
⚠️ spaCy unavailable – falling back to capitalisation heuristic
```
**Solution:** Download model
```bash
python -m spacy download en_core_web_sm
```

### No results returned
- Check query is not empty
- Verify database is loaded: `len(searcher._db) > 0`
- Lower `semantic_threshold` for broader matching
- Check `metadata["errors"]` for processing errors

### Slow semantic search
- Use `batch_semantic=True`
- Increase `top_k` to reduce redundant searches
- Clear cache if size exceeds 1,000 queries

---

## Implementation Details

### Normalization
Embeddings are L2-normalized so inner-product equals cosine similarity:
```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_normalized = embeddings / norms
```

### FAISS Indexing
- Flat index for <10K vectors
- IVF index with clustering for >10K vectors
- Metric: METRIC_INNER_PRODUCT

### Deduplication
Results are deduplicated by `(sign_id, query_gloss)` to prevent duplicate entries from exact + semantic matching.

---

## Related Files

- `main.py` - FastAPI wrapper
- `cloudbuild.yaml` - Cloud Build CI/CD config
- `deploy.sh` - Cloud Run deployment script
- `Dockerfile` - Container image

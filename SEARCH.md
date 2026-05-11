# KSL Semantic Search Documentation

## Overview

`KSLSearchFinal` is a semantic search engine for Kenyan Sign Language (KSL) gloss retrieval using SentenceTransformers and FAISS.

## Features

- Semantic similarity search
- FAISS cosine similarity indexing
- CSV + HuggingFace dataset support
- Duplicate filtering
- Cached FAISS index loading
- Fast top-k retrieval

## Installation

```bash
pip install sentence-transformers numpy pandas faiss-cpu datasets
```

## Quick Start

```python
from ksl_search_final import KSLSearchFinal

searcher = KSLSearchFinal()

searcher.load_data()

if searcher.index is None:
    searcher.build_index()

results = searcher.search("hello how are you", top_k=3)

for r in results:
    print(r)
```

## Query Format

- Input should be a normal English sentence
- Empty queries return empty results
- Search uses semantic similarity

## Example Output

```json
{
  "english": "hello",
  "ksl": "HELLO",
  "similarity_score": 0.92
}
```

## Notes

- Uses cosine similarity through normalized embeddings
- FAISS index is cached locally after build
- Duplicate English sentences are automatically removed

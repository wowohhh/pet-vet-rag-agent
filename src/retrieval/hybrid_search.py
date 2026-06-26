"""Hybrid retrieval: dense (vector) + sparse (BM25) with fusion."""

from rank_bm25 import BM25Okapi
from src.config import TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, BM25_WEIGHT
from src.retrieval.embeddings import embed_query
from src.retrieval.vector_store import query_collection, get_collection

_bm25_index: BM25Okapi | None = None
_bm25_docs: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for Chinese text (character-level bigrams)."""
    # For Chinese, character-level works better than whitespace splitting
    text = text.strip()
    tokens = []
    for i, ch in enumerate(text):
        tokens.append(ch)
        if i < len(text) - 1:
            tokens.append(ch + text[i + 1])
    return tokens


def build_bm25_index() -> int:
    """Build BM25 index from all chunks in ChromaDB.

    Returns:
        Number of documents indexed.
    """
    global _bm25_index, _bm25_docs

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])

    if not result["ids"]:
        _bm25_index = None
        _bm25_docs = []
        return 0

    _bm25_docs = []
    corpus = []
    for i, doc_id in enumerate(result["ids"]):
        text = result["documents"][i] if result["documents"] else ""
        meta = result["metadatas"][i] if result["metadatas"] else {}
        _bm25_docs.append({"id": doc_id, "text": text, "metadata": meta})
        corpus.append(_tokenize(text))

    _bm25_index = BM25Okapi(corpus)
    return len(corpus)


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> list[dict]:
    """Sparse retrieval using BM25.

    Args:
        query: Search query.
        top_k: Number of results.

    Returns:
        List of dicts with 'text', 'metadata', 'score'.
    """
    global _bm25_index, _bm25_docs

    if _bm25_index is None or not _bm25_docs:
        return []

    tokenized = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized)

    # Normalize scores
    max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "id": _bm25_docs[idx]["id"],
            "text": _bm25_docs[idx]["text"],
            "metadata": _bm25_docs[idx]["metadata"],
            "score": score / max_score,
        }
        for idx, score in ranked
    ]


def hybrid_search(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """Combined dense + sparse retrieval with score fusion.

    Uses weighted reciprocal rank fusion (RRF).

    Args:
        query: Search query.
        top_k: Number of final results.

    Returns:
        Ranked list of dicts with 'text', 'metadata', 'score', 'source'.
    """
    # Dense retrieval
    query_emb = embed_query(query)
    dense_results = query_collection(query_emb, top_k=TOP_K_VECTOR)

    # Sparse retrieval
    sparse_results = bm25_search(query, top_k=TOP_K_BM25)

    # Score fusion (weighted)
    combined: dict[str, dict] = {}

    # Add dense results
    for rank, r in enumerate(dense_results):
        key = r["id"]
        combined[key] = {
            "id": r["id"],
            "text": r["text"],
            "metadata": r["metadata"],
            "score": (1.0 - BM25_WEIGHT) * (1.0 / (rank + 60)),  # RRF-like
            "source": "dense",
        }

    # Add sparse results
    for rank, r in enumerate(sparse_results):
        key = r["id"]
        sparse_score = BM25_WEIGHT * (1.0 / (rank + 60))
        if key in combined:
            combined[key]["score"] += sparse_score
            combined[key]["source"] = "hybrid"
        else:
            combined[key] = {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": sparse_score,
                "source": "sparse",
            }

    # Sort by combined score
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

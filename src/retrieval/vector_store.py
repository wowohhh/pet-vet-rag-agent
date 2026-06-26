"""ChromaDB vector store operations."""

from __future__ import annotations

from typing import Any
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, CHROMA_COLLECTION
from src.retrieval.embeddings import embed_texts

_client: Any = None
_collection: Any = None


def get_client():
    """Get or initialize ChromaDB persistent client (singleton)."""
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    """Get or create the knowledge base collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"description": "Pet veterinary knowledge from CNKI papers"},
        )
    return _collection


def add_chunks(chunks: list[dict]) -> int:
    """Embed and add chunks to the vector store.

    Args:
        chunks: List of chunk dicts with 'text' and metadata fields.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{hash(c['text'])}_{c.get('chunk_index', 0)}" for c in chunks]
    embeddings = embed_texts(texts)

    # Extract metadata (non-text fields)
    metadatas = [
        {k: str(v) for k, v in c.items() if k != "text"}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def query_collection(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Query the vector store for similar chunks.

    Args:
        query_embedding: Query embedding vector.
        top_k: Number of results to return.

    Returns:
        List of dicts with 'text', 'metadata', and 'score'.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
            })
    return output


def get_chunk_count() -> int:
    """Return total number of chunks in the collection."""
    return get_collection().count()


def clear_collection() -> None:
    """Delete all chunks from the collection."""
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    global _collection
    _collection = None

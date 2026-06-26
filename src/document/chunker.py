"""Semantic text chunking for document ingestion."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, metadata: dict | None = None) -> list[dict]:
    """Split document text into overlapping chunks with metadata.

    Args:
        text: Cleaned text from a document.
        metadata: Document-level metadata (title, journal, year, etc.).

    Returns:
        List of dicts with 'text', 'chunk_index', and merged metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    base_meta = (metadata or {}).copy()
    return [
        {"text": chunk, "chunk_index": i, **base_meta}
        for i, chunk in enumerate(chunks)
    ]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk multiple documents.

    Args:
        documents: List of {'text': str, 'metadata': dict} from parser.

    Returns:
        Flat list of chunk dicts.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], metadata=doc.get("metadata", {}))
        all_chunks.extend(chunks)
    return all_chunks

"""Document ingestion pipeline: PDF → chunk → embed → ChromaDB."""

import sys
from pathlib import Path

from src.config import PAPERS_DIR
from src.document.parser import extract_text_from_pdf, extract_metadata
from src.document.chunker import chunk_text
from src.retrieval.vector_store import add_chunks, get_chunk_count, clear_collection
from src.retrieval.hybrid_search import build_bm25_index


def ingest_papers(papers_dir: Path | None = None, clear_first: bool = False):
    """Ingest all PDF papers into the knowledge base.

    Args:
        papers_dir: Directory containing CNKI PDF files.
        clear_first: If True, clear existing collection before ingesting.
    """
    papers_dir = papers_dir or PAPERS_DIR

    if not papers_dir.exists():
        print(f"错误: 论文目录不存在: {papers_dir}")
        print("请将 CNKI PDF 论文放入 data/papers/ 目录")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"错误: 未在 {papers_dir} 中找到 PDF 文件")
        return

    if clear_first:
        print("清空现有知识库...")
        clear_collection()

    print(f"找到 {len(pdf_files)} 篇论文")

    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i}/{len(pdf_files)}] 处理: {pdf_path.name}")

            # Extract text and metadata
            text = extract_text_from_pdf(pdf_path)
            meta = extract_metadata(pdf_path)

            # Chunk
            chunks = chunk_text(text, metadata=meta)
            print(f"  → {len(chunks)} 个文本块")

            # Add to vector store
            added = add_chunks(chunks)
            total_chunks += added
            print(f"  → 已存入 {added} 个块")

        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")

    # Build BM25 index
    print(f"\n总计: {total_chunks} 个文本块存入向量数据库")
    print("构建 BM25 索引...")
    bm25_count = build_bm25_index()
    print(f"BM25 索引: {bm25_count} 个文档")

    print(f"\n[OK] Import complete! Knowledge base has {get_chunk_count()} chunks.")


if __name__ == "__main__":
    import io
    import sys
    # Fix Windows GBK encoding for emoji/Unicode output
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    clear = "--clear" in sys.argv
    ingest_papers(clear_first=clear)

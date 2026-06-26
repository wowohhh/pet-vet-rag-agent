"""Central configuration for the Pet Vet RAG Agent."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"

# ChromaDB — must use English-only path (rust backend can't handle Chinese chars)
CHROMA_DIR = Path("C:/rag_data/chroma_db")
CHROMA_COLLECTION = "pet_vet_knowledge"

# Models
# Use ModelScope downloaded model (offline, no HF needed)
EMBEDDING_MODEL = str(PROJECT_ROOT / "data" / "models" / "BAAI" / "bge-small-zh-v1___5")
EMBEDDING_DEVICE = "cuda"  # or "cpu"

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")  # 4B fits 8GB VRAM

# Document processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_FINAL = 5
BM25_WEIGHT = 0.3  # BM25 contribution in hybrid fusion

# Agent
MAX_ITERATIONS = 5  # ReAct agent max cycles

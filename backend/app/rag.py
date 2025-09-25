from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .settings import settings

def _resolve_chroma_path() -> Path:
    """
    If CHROMA_DIR is absolute, use it.
    If relative, resolve it against the project root (two levels up from this file).
    """
    raw = Path(settings.chroma_dir)
    if raw.is_absolute():
        return raw
    project_root = Path(__file__).resolve().parents[2]  # .../ai-medical-chatbot
    return (project_root / raw).resolve()

class RAGPipeline:
    def __init__(self):
        storage_path = _resolve_chroma_path()
        storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=ChromaSettings()
        )
        self.collection = self.client.get_or_create_collection(settings.collection_name)
        self.embedder = SentenceTransformer(settings.embedding_model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # normalize for cosine similarity
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def add_texts(self, ids: List[str], texts: List[str], metadatas: List[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        embs = self.embed(texts)
        self.collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)

    def query(self, question: str, k: int = 5):
        """Return (ids, docs, metas, sims)."""
        q_emb = self.embed([question])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        sims = [1.0 / (1.0 + float(d)) for d in dists]
        return ids, docs, metas, sims

# Singleton instance for app use
rag = RAGPipeline()


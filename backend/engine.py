"""
engine.py
---------
Core SkillSync matching engine.

Indexes developer code snippets into a persistent ChromaDB collection
and matches natural-language task descriptions to the most relevant
developers using semantic (cosine) similarity.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PERSIST_DIR = "data/chroma_db"
CODE_COLLECTION = "developer_skills"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SkillSyncEngine:
    """Semantic skill-matching engine backed by ChromaDB + sentence-transformers."""

    def __init__(self) -> None:
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=CODE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_developer_code(
        self, dev_name: str, code_snippet: str, repo_name: str
    ) -> None:
        """Index a developer's code snippet into ChromaDB."""
        embedding = self.model.encode(
            code_snippet, normalize_embeddings=True
        ).tolist()
        doc_id = f"{dev_name}_{abs(hash(code_snippet))}"

        self.collection.upsert(
            documents=[code_snippet],
            embeddings=[embedding],
            metadatas=[{"developer": dev_name, "repo": repo_name}],
            ids=[doc_id],
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def find_best_match(
        self, task_description: str, n_results: int = 1
    ) -> dict[str, Any] | None:
        """Return the single best-matching developer for *task_description*."""
        results = self.query_developers(task_description, n_results=n_results)
        return results[0] if results else None

    def query_developers(
        self, task_description: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Return up to *n_results* developers ranked by semantic similarity
        to *task_description*.

        Each result dict contains:
            developer (str), repo (str), code_snippet (str), score (float 0-1)
        """
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.model.encode(
            task_description, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["metadatas", "distances", "documents"],
        )

        matches = []
        for meta, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            matches.append(
                {
                    "developer": meta["developer"],
                    "repo": meta["repo"],
                    "code_snippet": doc,
                    # cosine distance → similarity (higher = better match)
                    "score": round(1 - dist, 4),
                }
            )
        return matches

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of indexed code snippets."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete and recreate the code-skills collection."""
        self.chroma_client.delete_collection(CODE_COLLECTION)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CODE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

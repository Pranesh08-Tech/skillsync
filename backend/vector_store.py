"""
vector_store.py
---------------
Manages ChromaDB collections for storing and querying contributor
skill embeddings.  Uses sentence-transformers to produce embeddings
locally so no external embedding API key is required.
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "contributor_skills"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, 384-dim, MIT licence

# ---------------------------------------------------------------------------
# Singletons (lazy-initialised)
# ---------------------------------------------------------------------------

_client: chromadb.ClientAPI | None = None
_model: SentenceTransformer | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Return a normalised embedding vector for *text*."""
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def upsert_contributor(contributor: dict[str, Any]) -> None:
    """
    Insert or update a contributor record in the vector store.

    Expected keys in *contributor*:
        login           (str)  – GitHub username, used as document ID
        contributions   (int)  – commit count
        repo_languages  (list) – languages detected in the repo
        Any extra keys are stored as metadata.
    """
    login: str = contributor["login"]
    languages: list[str] = contributor.get("repo_languages", [])
    contributions: int = contributor.get("contributions", 0)

    # Build a plain-text description for embedding
    skill_text = (
        f"GitHub contributor {login} has {contributions} contributions. "
        f"Languages used: {', '.join(languages) if languages else 'unknown'}."
    )

    embedding = embed_text(skill_text)
    metadata = {
        "login": login,
        "contributions": contributions,
        "languages": json.dumps(languages),
        "avatar_url": contributor.get("avatar_url", ""),
        "html_url": contributor.get("html_url", ""),
    }

    _get_collection().upsert(
        ids=[login],
        embeddings=[embedding],
        documents=[skill_text],
        metadatas=[metadata],
    )


def upsert_contributors(contributors: list[dict[str, Any]]) -> None:
    """Batch upsert a list of contributor records."""
    for contributor in contributors:
        upsert_contributor(contributor)


def query_similar_contributors(
    query: str, n_results: int = 5
) -> list[dict[str, Any]]:
    """
    Return the *n_results* contributors whose skill profiles are most
    semantically similar to *query*.

    Each result dict contains:
        login, contributions, languages (list), score (float 0-1),
        avatar_url, html_url, document
    """
    embedding = embed_text(query)
    collection = _get_collection()

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, count),
        include=["metadatas", "distances", "documents"],
    )

    output = []
    for meta, dist, doc in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        output.append(
            {
                "login": meta["login"],
                "contributions": meta["contributions"],
                "languages": json.loads(meta.get("languages", "[]")),
                "score": round(1 - dist, 4),  # cosine distance → similarity
                "avatar_url": meta.get("avatar_url", ""),
                "html_url": meta.get("html_url", ""),
                "document": doc,
            }
        )
    return output


def list_all_contributors() -> list[dict[str, Any]]:
    """Return all stored contributor metadata records."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas", "documents"])
    output = []
    for meta, doc in zip(results["metadatas"], results["documents"]):
        output.append(
            {
                "login": meta["login"],
                "contributions": meta["contributions"],
                "languages": json.loads(meta.get("languages", "[]")),
                "avatar_url": meta.get("avatar_url", ""),
                "html_url": meta.get("html_url", ""),
                "document": doc,
            }
        )
    return output


def clear_collection() -> None:
    """Delete and recreate the contributors collection (dev/test helper)."""
    client = _get_client()
    client.delete_collection(COLLECTION_NAME)
    _get_collection()  # recreates it

"""Access to the local ChromaDB index, shared by the pipeline and the scripts."""

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "vector_db"
COLLECTION = "medical"


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    topic: str
    section: str
    source: str
    url: str
    distance: float

    @property
    def body(self) -> str:
        return self.text.split("\n", 1)[-1]


def group_key(group: str) -> str:
    """Metadata field name for a topic group, e.g. 'Blood, Heart and Circulation'.

    Chroma metadata holds single values only, so every group a page belongs to
    is stored as its own true/false field.
    """
    return "group_" + re.sub(r"[^a-z0-9]+", "_", group.lower()).strip("_")


def group_filter(groups) -> dict:
    """Chroma 'where' filter matching chunks in any of the given groups."""
    clauses = [{group_key(group): True} for group in groups]
    if not clauses:
        raise ValueError("No topic groups given: an empty selection must match nothing.")
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def open_client():
    """Persistent client on vector_db/ with telemetry off (everything stays local)."""
    return chromadb.PersistentClient(
        path=str(INDEX_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def open_collection():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"The document index was not found at {INDEX_DIR}. "
            "Build it first with: python scripts/build_index.py"
        )
    try:
        return open_client().get_collection(COLLECTION)
    except Exception as error:
        raise RuntimeError(
            f"The '{COLLECTION}' collection could not be opened. "
            "Rebuild the index with: python scripts/build_index.py"
        ) from error


def search(collection, question: str, groups=None, k: int = 5, max_distance: float = float("inf")):
    """Return up to k chunks closest to the question, nearest first.

    groups=None searches everything. Chunks farther than max_distance are dropped
    so unrelated questions come back empty instead of with the "least bad" text.
    """
    where = group_filter(groups) if groups is not None else None
    result = collection.query(query_texts=[question], n_results=k, where=where)
    chunks = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if distance <= max_distance:
            chunks.append(
                RetrievedChunk(
                    text=text,
                    topic=meta["topic"],
                    section=meta["section"],
                    source=meta["source"],
                    url=meta["url"],
                    distance=distance,
                )
            )
    return chunks

"""Retrieve the most relevant CV sections from ChromaDB for a given job description.

Uses sentence-transformers (all-MiniLM-L6-v2) embeddings — the same model
that embed_cv.py uses — to find top-k CV chunks matching the role and skills
extracted by jd_parser. Template and instructions chunks are filtered out;
those are reserved for resume_tailor only.

No LLM calls — pure embedding retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import CHROMA_STORE_DIR

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "master_cv"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Metadata types to exclude — these are for resume_tailor, not for
# match scoring or generation.
_EXCLUDED_TYPES = {"template", "instructions"}

# Lazy singletons — initialised on first call, not at import time.
_chroma_collection = None
_embedder = None


def _get_embedder():
    """Lazily load the sentence-transformers embedding model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: %s", _EMBEDDING_MODEL)
        _embedder = SentenceTransformer(_EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    """Lazily initialise the ChromaDB client and return the master_cv collection."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        chroma_path = Path(CHROMA_STORE_DIR)
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB store not found at {chroma_path}. "
                "Run 'python cv/embed_cv.py' first to embed your CV."
            )

        client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_collection = client.get_collection(name=_COLLECTION_NAME)
        logger.info(
            "Connected to ChromaDB collection '%s' (%d documents)",
            _COLLECTION_NAME,
            _chroma_collection.count(),
        )
    return _chroma_collection


def _build_query(requirements: dict) -> str:
    """Build a query string from the requirements dict for embedding similarity search.

    Combines role and skills into a single natural-language query that
    will produce a useful embedding for cosine similarity matching.
    """
    parts: list[str] = []

    role = requirements.get("role")
    if role:
        parts.append(role)

    skills = requirements.get("skills", [])
    if skills:
        parts.append(" ".join(skills))

    query = " ".join(parts).strip()
    if not query:
        logger.warning("Empty query built from requirements — using fallback 'software engineer'")
        query = "software engineer"

    return query


def retrieve_cv_sections(requirements: dict, top_k: int = 4) -> list[str]:
    """Retrieve the most relevant CV sections from ChromaDB for the given requirements.

    Args:
        requirements: The requirements dict from jd_parser.parse_jd().
                      Must contain 'role' (str|None) and 'skills' (list[str]).
        top_k: Maximum number of sections to return. Defaults to 4.

    Returns:
        A list of CV section text strings, ordered by relevance (most
        relevant first). May return fewer than top_k if the collection
        has fewer CV section chunks.
    """
    query_text = _build_query(requirements)
    logger.info("CV RAG query (%d chars): %.120s", len(query_text), query_text)

    # Embed the query
    embedder = _get_embedder()
    query_embedding = embedder.encode(query_text).tolist()

    # Fetch more results than needed so we can filter out template/instructions
    # and still return up to top_k CV sections.
    collection = _get_collection()
    total_docs = collection.count()
    n_results = min(total_docs, top_k + len(_EXCLUDED_TYPES) + 2)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack — ChromaDB returns lists-of-lists (one per query).
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Filter out template and instructions chunks, keep only cv_section.
    sections: list[str] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunk_type = meta.get("type", "cv_section")
        if chunk_type in _EXCLUDED_TYPES:
            logger.debug(
                "Skipping chunk type='%s' (section='%s')",
                chunk_type,
                meta.get("section_name", "?"),
            )
            continue

        sections.append(doc)
        logger.info(
            "  [%.4f] %s (%d chars)",
            dist,
            meta.get("section_name", "unknown"),
            len(doc),
        )

        if len(sections) >= top_k:
            break

    logger.info(
        "CV RAG returned %d sections for query: %.80s",
        len(sections),
        query_text,
    )

    if not sections:
        logger.warning(
            "No CV sections retrieved — ChromaDB may be empty or all chunks are "
            "template/instructions. Run 'python cv/embed_cv.py' to re-embed."
        )

    return sections

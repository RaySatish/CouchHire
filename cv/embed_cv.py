"""
cv/embed_cv.py — Orchestrates the full CV embed pipeline.

Steps:
1. Find master CV in cv/uploads/ (.tex, .pdf, or .docx — priority order)
2. Parse into sections via cv_parser.parse_cv()
3. Load resume template (user's or default)
4. Load tailoring instructions (user's or default)
5. Embed each section using sentence-transformers (all-MiniLM-L6-v2)
6. Store in ChromaDB at cv/chroma_store/ with metadata
7. Store template and instructions as special chunks
8. Print summary

Re-running clears old collection and re-embeds fresh.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Paths ───
_CV_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CV_DIR.parent
_UPLOADS_DIR = _CV_DIR / "uploads"
_DEFAULTS_DIR = _CV_DIR / "defaults"
_CHROMA_DIR = _CV_DIR / "chroma_store"
_COLLECTION_NAME = "master_cv"

# Priority order for master CV discovery
_CV_EXTENSIONS = [".tex", ".pdf", ".docx"]


def _find_master_cv() -> Path:
    """Find the master CV in uploads/ directory. Returns path or exits with instructions."""
    for ext in _CV_EXTENSIONS:
        candidate = _UPLOADS_DIR / f"master_cv{ext}"
        if candidate.exists():
            logger.info("Found master CV: %s", candidate)
            return candidate

    print("\n" + "=" * 60)
    print("ERROR: No master CV found in cv/uploads/")
    print("=" * 60)
    print()
    print("Place your master CV in the cv/uploads/ directory as one of:")
    for ext in _CV_EXTENSIONS:
        print(f"  cv/uploads/master_cv{ext}")
    print()
    print("Then re-run: python cv/embed_cv.py")
    print("=" * 60 + "\n")
    sys.exit(1)


def _load_text_file(user_path: Path, default_path: Path, label: str) -> str:
    """Load a text file from user path, falling back to default."""
    if user_path.exists():
        logger.info("Using user %s: %s", label, user_path)
        return user_path.read_text(encoding="utf-8")
    elif default_path.exists():
        logger.info("Using default %s: %s", label, default_path)
        return default_path.read_text(encoding="utf-8")
    else:
        logger.warning("No %s found at %s or %s", label, user_path, default_path)
        return ""


def _get_embedder():
    """Load sentence-transformers model. Returns the model instance."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def _get_chroma_collection():
    """Get or create the ChromaDB collection, clearing any existing data."""
    import chromadb

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

    # Delete existing collection for fresh re-embed
    # ChromaDB may raise ValueError or NotFoundError depending on version
    existing_names = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME in existing_names:
        client.delete_collection(_COLLECTION_NAME)
        logger.info("Cleared existing ChromaDB collection: %s", _COLLECTION_NAME)

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def run_embed() -> None:
    """Run the full embed pipeline: find CV -> parse -> embed -> store."""
    # Ensure project root is on sys.path so cv.cv_parser is importable
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Step 1: Find master CV
    cv_path = _find_master_cv()
    source_format = cv_path.suffix.lstrip(".")

    # Step 2: Parse into sections
    from cv.cv_parser import parse_cv
    sections = parse_cv(cv_path)

    if not sections:
        print("ERROR: No sections parsed from CV. Check file content.")
        sys.exit(1)

    # Step 3: Load resume template
    template_text = _load_text_file(
        _UPLOADS_DIR / "resume_template.tex",
        _DEFAULTS_DIR / "resume_template.tex",
        "resume template",
    )

    # Step 4: Load tailoring instructions
    instructions_text = _load_text_file(
        _UPLOADS_DIR / "instructions.md",
        _DEFAULTS_DIR / "instructions.md",
        "tailoring instructions",
    )

    # Step 5: Load embedder
    print("Loading sentence-transformers model...")
    embedder = _get_embedder()

    # Step 6: Prepare chunks for embedding
    timestamp = datetime.now(timezone.utc).isoformat()
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for section_name, section_text in sections.items():
        documents.append(section_text)
        metadatas.append({
            "section_name": section_name,
            "char_count": len(section_text),
            "source_format": source_format,
            "timestamp": timestamp,
            "type": "cv_section",
        })
        ids.append(f"cv_{section_name.lower().replace(' ', '_')}")

    # Step 7: Add template and instructions as special chunks
    if template_text:
        documents.append(template_text)
        metadatas.append({
            "section_name": "resume_template",
            "char_count": len(template_text),
            "source_format": "tex",
            "timestamp": timestamp,
            "type": "template",
        })
        ids.append("special_template")

    if instructions_text:
        documents.append(instructions_text)
        metadatas.append({
            "section_name": "tailoring_instructions",
            "char_count": len(instructions_text),
            "source_format": "md",
            "timestamp": timestamp,
            "type": "instructions",
        })
        ids.append("special_instructions")

    # Embed all documents
    print(f"Embedding {len(documents)} chunks...")
    embeddings = embedder.encode(documents, show_progress_bar=False).tolist()

    # Store in ChromaDB
    collection = _get_chroma_collection()
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Step 8: Print summary
    print()
    print("=" * 50)
    print("CV Embed Pipeline — Complete")
    print("=" * 50)
    print(f"  Source:       {cv_path}")
    print(f"  Format:       {source_format}")
    print(f"  Sections:     {len(sections)}")
    for name in sections:
        print(f"    - {name}")
    print(f"  Template:     {'loaded' if template_text else 'NOT FOUND'}")
    print(f"  Instructions: {'loaded' if instructions_text else 'NOT FOUND'}")
    print(f"  Total chunks: {len(documents)}")
    print(f"  ChromaDB:     {_CHROMA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
    run_embed()

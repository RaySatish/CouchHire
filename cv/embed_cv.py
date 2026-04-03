"""Orchestrates the full CV embed pipeline.

Finds the master CV, parses it into sections, embeds each section
using sentence-transformers, and stores everything in ChromaDB.
Also stores the resume template and tailoring instructions as
special chunks for retrieval by resume_tailor.py.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve paths relative to this file
_CV_DIR = Path(__file__).resolve().parent
_UPLOADS_DIR = _CV_DIR / "uploads"
_DEFAULTS_DIR = _CV_DIR / "defaults"
_CHROMA_DIR = _CV_DIR / "chroma_store"
_COLLECTION_NAME = "master_cv"

# Supported CV formats in priority order
_CV_FORMATS = [".tex", ".pdf", ".docx"]


def _find_master_cv() -> Path | None:
    """Find the master CV in uploads/ — tries .tex, .pdf, .docx in order."""
    for ext in _CV_FORMATS:
        candidate = _UPLOADS_DIR / f"master_cv{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_text_file(uploads_name: str, defaults_name: str) -> tuple[str, str]:
    """Load a text file from uploads/ with fallback to defaults/.

    Returns (content, source_label) where source_label is 'uploads' or 'defaults'.
    """
    uploads_path = _UPLOADS_DIR / uploads_name
    defaults_path = _DEFAULTS_DIR / defaults_name

    if uploads_path.exists():
        return uploads_path.read_text(encoding="utf-8"), "uploads"
    elif defaults_path.exists():
        return defaults_path.read_text(encoding="utf-8"), "defaults"
    else:
        raise FileNotFoundError(
            f"Neither {uploads_path} nor {defaults_path} found. "
            f"Please provide at least the default file."
        )


def embed() -> None:
    """Run the full embed pipeline: find CV → parse → embed → store."""
    # ── Step 1: Find master CV ──
    cv_path = _find_master_cv()
    if cv_path is None:
        print("\n❌ No master CV found in cv/uploads/")
        print("   Place one of these files there:")
        print("     • cv/uploads/master_cv.tex")
        print("     • cv/uploads/master_cv.pdf")
        print("     • cv/uploads/master_cv.docx")
        print("\n   Then re-run: python cv/embed_cv.py")
        sys.exit(1)

    print(f"📄 Found master CV: {cv_path.name}")

    # ── Step 2: Parse into sections ──
    from cv.cv_parser import parse_cv
    sections = parse_cv(cv_path)
    if not sections:
        print("❌ No sections detected in CV. Check formatting.")
        sys.exit(1)

    print(f"📑 Parsed {len(sections)} sections: {list(sections.keys())}")

    # ── Step 3: Load template and instructions ──
    template_text, template_source = _load_text_file(
        "resume_template.tex", "resume_template.tex"
    )
    instructions_text, instructions_source = _load_text_file(
        "instructions.md", "instructions.md"
    )

    print(f"📝 Template loaded from: {template_source}")
    print(f"📝 Instructions loaded from: {instructions_source}")

    # ── Step 4: Set up embeddings ──
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    try:
        import chromadb
    except ImportError:
        print("❌ chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Step 5: Set up ChromaDB ──
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

    # Delete old collection if it exists (re-running clears and re-embeds)
    try:
        client.delete_collection(_COLLECTION_NAME)
        print("🔄 Cleared existing embeddings")
    except ValueError:
        pass  # Collection didn't exist

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    source_format = cv_path.suffix.lstrip(".")

    # ── Step 6: Embed CV sections ──
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
        vec = model.encode(section_text).tolist()
        doc_id = f"cv_{section_name.lower().replace(' ', '_')}"

        ids.append(doc_id)
        documents.append(section_text)
        embeddings.append(vec)
        metadatas.append({
            "type": "cv_section",
            "section_name": section_name,
            "char_count": len(section_text),
            "source_format": source_format,
            "timestamp": timestamp,
        })

    # ── Step 7: Embed template and instructions as special chunks ──
    template_vec = model.encode(template_text).tolist()
    ids.append("resume_template")
    documents.append(template_text)
    embeddings.append(template_vec)
    metadatas.append({
        "type": "template",
        "section_name": "resume_template",
        "char_count": len(template_text),
        "source_format": "tex",
        "timestamp": timestamp,
    })

    instructions_vec = model.encode(instructions_text).tolist()
    ids.append("tailoring_instructions")
    documents.append(instructions_text)
    embeddings.append(instructions_vec)
    metadatas.append({
        "type": "instructions",
        "section_name": "tailoring_instructions",
        "char_count": len(instructions_text),
        "source_format": "md",
        "timestamp": timestamp,
    })

    # ── Step 8: Add all to ChromaDB ──
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # ── Step 9: Print summary ──
    cv_count = len(sections)
    total_count = collection.count()
    print(f"\n✅ Embedding complete!")
    print(f"   CV sections embedded: {cv_count}")
    print(f"   Template chunk: 1 (from {template_source})")
    print(f"   Instructions chunk: 1 (from {instructions_source})")
    print(f"   Total chunks in ChromaDB: {total_count}")
    print(f"   Store location: {_CHROMA_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
    embed()

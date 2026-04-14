#!/usr/bin/env bash
# =============================================================================
# CouchHire — Docker entrypoint
# Ensures ChromaDB embeddings exist before starting the main process.
# =============================================================================
set -e

CHROMA_DIR="/app/cv/chroma_store"
UPLOADS_DIR="/app/cv/uploads"

# If chroma_store is empty/missing AND uploads exist, run embed pipeline
if [ ! -f "$CHROMA_DIR/chroma.sqlite3" ] && [ -d "$UPLOADS_DIR" ] && [ "$(ls -A $UPLOADS_DIR 2>/dev/null)" ]; then
    echo "[entrypoint] ChromaDB store not found — running embed_cv.py..."
    python cv/embed_cv.py
    echo "[entrypoint] Embedding complete."
else
    if [ -f "$CHROMA_DIR/chroma.sqlite3" ]; then
        echo "[entrypoint] ChromaDB store found — skipping embed."
    else
        echo "[entrypoint] No CV uploads found — skipping embed (add files to cv/uploads/ and restart)."
    fi
fi

# Execute the CMD passed to the container
exec "$@"

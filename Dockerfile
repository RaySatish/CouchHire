# =============================================================================
# CouchHire — Dockerfile
# Multi-purpose image: pipeline, dashboard, telegram bot
# =============================================================================

FROM python:3.11-slim AS base

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Force headless browser inside container
    BROWSER_HEADLESS=true

WORKDIR /app

# ---- System dependencies ----------------------------------------------------
# texlive for pdflatex, chromium for browser agent, misc utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    chromium \
    chromium-driver \
    # Playwright system deps (subset — playwright install --with-deps adds the rest)
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libcups2 \
    libxss1 \
    libgtk-3-0 \
    # General utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Playwright Chromium -----------------------------------------------------
RUN playwright install chromium --with-deps 2>/dev/null || playwright install chromium

# ---- spaCy model -------------------------------------------------------------
RUN python -m spacy download en_core_web_sm

# ---- Copy project files ------------------------------------------------------
COPY . .

# ---- Entrypoint script -------------------------------------------------------
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command — override in docker-compose per service
CMD ["python", "pipeline.py", "--help"]

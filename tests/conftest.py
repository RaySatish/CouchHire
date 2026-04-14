"""Shared pytest fixtures for CouchHire test suite.

Provides mock Supabase client, mock LLM responses, sample JD text,
sample PipelineState, and temporary output directories.  All external
services are mocked so tests run fully offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Environment stubs — set BEFORE any config import
# ---------------------------------------------------------------------------
_ENV_DEFAULTS = {
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": "test-key-groq",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "test-supabase-key",
    "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF",
    "TELEGRAM_CHAT_ID": "12345678",
    "GMAIL_MCP_URL": "http://localhost:8000/mcp",
    "GITHUB_URL": "https://github.com/testuser",
    "APPLICANT_NAME": "Test User",
    "APPLICANT_EMAIL": "test@example.com",
    "APPLICANT_PHONE": "+1-555-0100",
    "APPLICANT_LINKEDIN": "https://linkedin.com/in/testuser",
    "MATCH_THRESHOLD": "60",
}

for key, val in _ENV_DEFAULTS.items():
    os.environ.setdefault(key, val)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_JD_TEXT = """\
Role: Machine Learning Engineer (New Grad / Entry Level)

Company: QuantumLeap AI

Location: Bangalore, India (Hybrid)

About the Role:
QuantumLeap AI is building next-generation recommendation systems.
We're looking for a Machine Learning Engineer to join our Applied ML team.

Responsibilities:
- Design and implement ML models for recommendation systems
- Build end-to-end ML pipelines
- Implement deep learning models (LSTM, Transformer, CNN)

Requirements:
- B.Tech/M.Tech in CS, Statistics, or related field
- Strong Python programming skills
- Experience with PyTorch or TensorFlow
- Knowledge of NLP, computer vision, or recommender systems
- Familiarity with Docker, Kubernetes, MLflow

Apply: Send your resume to hr@quantumleap.ai with subject "ML Engineer Application - [Your Name]"
"""

SAMPLE_JD_EMAIL = """\
Role: Backend Developer
Company: Acme Corp
Apply by emailing jobs@acme.com with your resume attached.
Skills: Python, Django, PostgreSQL, Docker
"""

SAMPLE_JD_URL = """\
Role: Frontend Engineer
Company: Widget Inc
Apply at: https://jobs.lever.co/widget/12345
Skills: React, TypeScript, CSS, GraphQL
"""

SAMPLE_JD_MANUAL = """\
Role: Data Analyst
Company: DataCo
We are hiring a data analyst. Check our careers page for details.
Skills: SQL, Python, Tableau, Excel
"""

SAMPLE_REQUIREMENTS = {
    "company": "QuantumLeap AI",
    "role": "Machine Learning Engineer",
    "skills": ["Python", "PyTorch", "TensorFlow", "NLP", "Docker"],
    "apply_method": "email",
    "apply_target": "hr@quantumleap.ai",
    "cover_letter_required": False,
    "subject_line_format": "ML Engineer Application - [Your Name]",
    "email_instructions": None,
    "github_requested": False,
    "form_fields": [],
}

SAMPLE_CV_SECTIONS = [
    "Experience: Built a recommendation engine using PyTorch and collaborative filtering. "
    "Deployed on AWS with Docker containers and CI/CD pipelines.",
    "Projects: Maritime Situational Awareness system using OCR and RAG pipelines. "
    "NIFTY50 portfolio optimizer combining LSTM forecasting with FinBERT sentiment analysis.",
    "Skills: Python, PyTorch, TensorFlow, scikit-learn, Docker, Kubernetes, MLflow, SQL",
    "Education: M.Sc. Computational Statistics and Data Analytics, VIT Vellore (CGPA: 9.23)",
]

SAMPLE_RESUME_CONTENT = (
    "EXPERIENCE: Built recommendation engine with PyTorch and collaborative filtering. "
    "Deployed on AWS with Docker. "
    "PROJECTS: Maritime Situational Awareness (OCR + RAG). "
    "NIFTY50 portfolio optimizer (LSTM + FinBERT). "
    "SKILLS: Python, PyTorch, TensorFlow, scikit-learn, Docker, Kubernetes, MLflow."
)


@pytest.fixture
def sample_jd_text() -> str:
    """Return a sample job description text."""
    return SAMPLE_JD_TEXT


@pytest.fixture
def sample_jd_email() -> str:
    """Return a sample JD with email apply method."""
    return SAMPLE_JD_EMAIL


@pytest.fixture
def sample_jd_url() -> str:
    """Return a sample JD with URL apply method."""
    return SAMPLE_JD_URL


@pytest.fixture
def sample_jd_manual() -> str:
    """Return a sample JD with manual apply method."""
    return SAMPLE_JD_MANUAL


@pytest.fixture
def sample_requirements() -> dict:
    """Return a sample parsed requirements dict."""
    return dict(SAMPLE_REQUIREMENTS)


@pytest.fixture
def sample_cv_sections() -> list[str]:
    """Return sample CV sections."""
    return list(SAMPLE_CV_SECTIONS)


@pytest.fixture
def sample_resume_content() -> str:
    """Return sample resume content summary."""
    return SAMPLE_RESUME_CONTENT


@pytest.fixture
def sample_pipeline_state(sample_requirements, sample_cv_sections) -> dict:
    """Return a sample PipelineState dict with all key fields populated."""
    return {
        "jd_text": SAMPLE_JD_TEXT,
        "jd_url": None,
        "job_url_direct": None,
        "source": "cli",
        "requirements": sample_requirements,
        "cv_sections": sample_cv_sections,
        "match_score": 75.5,
        "resume_pdf_path": "/tmp/test_resume.pdf",
        "resume_content": SAMPLE_RESUME_CONTENT,
        "cover_letter_text": None,
        "cover_letter_pdf_path": None,
        "email_subject": "Application for ML Engineer — Test User",
        "email_body": "Hi,\n\nI'm applying for the ML Engineer role...",
        "route": "email",
        "draft_url": "https://mail.google.com/mail/u/0/#drafts/abc123",
        "draft_id": "abc123",
        "form_result": None,
        "application_id": "test-uuid-1234",
        "_approval": None,
        "_rejection_reason": None,
        "_gate2_approval": None,
        "_gate2_rejection_reason": None,
        "email_sent": False,
        "sent_url": None,
        "error": None,
    }


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Return a temporary directory for test output files."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Mock fixtures for external services
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_complete():
    """Mock llm.client.complete() to return a canned response."""
    with patch("llm.client.complete") as mock_fn:
        mock_fn.return_value = json.dumps(SAMPLE_REQUIREMENTS)
        yield mock_fn


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client with CRUD stubs."""
    mock_client = MagicMock()

    # insert returns a row with id
    mock_response = MagicMock()
    mock_response.data = [{"id": "test-uuid-1234", "company": "TestCo", "role": "Engineer"}]
    mock_client.table.return_value.insert.return_value.execute.return_value = mock_response

    # update returns updated row
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

    # select returns list
    mock_client.table.return_value.select.return_value.order.return_value.range.return_value.execute.return_value = mock_response
    mock_client.table.return_value.select.return_value.order.return_value.execute.return_value = mock_response
    mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(data={"id": "test-uuid-1234"})

    # not_ chain for get_labeled_outcomes
    mock_client.table.return_value.select.return_value.not_.is_.return_value.order.return_value.execute.return_value = mock_response

    with patch("db.supabase_client._get_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB collection."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "documents": [SAMPLE_CV_SECTIONS],
        "metadatas": [[
            {"type": "cv_section", "section_name": "experience"},
            {"type": "cv_section", "section_name": "projects"},
            {"type": "cv_section", "section_name": "skills"},
            {"type": "cv_section", "section_name": "education"},
        ]],
        "distances": [[0.2, 0.3, 0.4, 0.5]],
    }
    return mock_collection


@pytest.fixture
def mock_telegram():
    """Mock Telegram bot send functions."""
    with patch("bot.telegram_bot.send_notification") as mock_notify, \
         patch("bot.telegram_bot.send_document") as mock_doc, \
         patch("bot.telegram_bot.send_photo") as mock_photo:
        yield {
            "notify": mock_notify,
            "document": mock_doc,
            "photo": mock_photo,
        }


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading real models."""
    import numpy as np
    mock_model = MagicMock()
    # encode returns a numpy array of shape (embedding_dim,) or (n, embedding_dim)
    def _encode(texts, **kwargs):
        if isinstance(texts, str):
            return np.random.rand(384).astype(np.float32)
        return np.random.rand(len(texts), 384).astype(np.float32)
    mock_model.encode = _encode
    return mock_model


@pytest.fixture
def sample_cover_letter_text() -> str:
    """Return a sample cover letter text."""
    return (
        "QuantumLeap AI's work on next-generation recommendation systems "
        "represents exactly the kind of applied ML challenge I've been "
        "preparing for throughout my academic career.\n\n"
        "My experience building a recommendation engine using PyTorch and "
        "collaborative filtering taught me that the real challenge isn't "
        "choosing the right architecture—it's understanding the problem "
        "deeply enough to know which signals matter.\n\n"
        "I'd welcome the chance to discuss how my background in applied ML "
        "aligns with your team's goals. I look forward to hearing from you."
    )


@pytest.fixture
def sample_email_requirements() -> dict:
    """Return requirements dict for email route."""
    return {
        "company": "Acme Corp",
        "role": "Backend Developer",
        "skills": ["Python", "Django", "PostgreSQL"],
        "apply_method": "email",
        "apply_target": "jobs@acme.com",
        "cover_letter_required": False,
        "subject_line_format": None,
        "email_instructions": None,
        "github_requested": False,
        "form_fields": [],
    }


@pytest.fixture
def sample_url_requirements() -> dict:
    """Return requirements dict for form/URL route."""
    return {
        "company": "Widget Inc",
        "role": "Frontend Engineer",
        "skills": ["React", "TypeScript", "CSS"],
        "apply_method": "url",
        "apply_target": "https://jobs.lever.co/widget/12345",
        "cover_letter_required": False,
        "subject_line_format": None,
        "email_instructions": None,
        "github_requested": False,
        "form_fields": [],
    }


@pytest.fixture
def sample_manual_requirements() -> dict:
    """Return requirements dict for manual route."""
    return {
        "company": "DataCo",
        "role": "Data Analyst",
        "skills": ["SQL", "Python", "Tableau"],
        "apply_method": "unknown",
        "apply_target": None,
        "cover_letter_required": False,
        "subject_line_format": None,
        "email_instructions": None,
        "github_requested": False,
        "form_fields": [],
    }


# ---------------------------------------------------------------------------
# Load sample JDs from test_jds/ directory
# ---------------------------------------------------------------------------

_TEST_JDS_DIR = PROJECT_ROOT / "tests" / "test_jds"


def _load_test_jds() -> list[tuple[str, str]]:
    """Load all .txt JD files from test_jds/ as (filename, content) tuples."""
    jds = []
    if _TEST_JDS_DIR.exists():
        for f in sorted(_TEST_JDS_DIR.glob("*.txt")):
            jds.append((f.name, f.read_text(encoding="utf-8")))
    return jds


TEST_JDS = _load_test_jds()


@pytest.fixture(params=TEST_JDS, ids=[t[0] for t in TEST_JDS] if TEST_JDS else [])
def test_jd_file(request) -> tuple[str, str]:
    """Parametrized fixture yielding (filename, content) for each test JD."""
    return request.param

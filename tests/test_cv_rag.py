"""Tests for agents/cv_rag.py — ChromaDB retrieval with mocked dependencies."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from agents.cv_rag import _build_query, retrieve_cv_sections


# ── _build_query ──────────────────────────────────────────────────────────

class TestBuildQuery:
    def test_role_and_skills(self):
        reqs = {"role": "ML Engineer", "skills": ["Python", "PyTorch"]}
        result = _build_query(reqs)
        assert "ML Engineer" in result
        assert "Python" in result
        assert "PyTorch" in result

    def test_role_only(self):
        reqs = {"role": "Backend Developer", "skills": []}
        result = _build_query(reqs)
        assert "Backend Developer" in result

    def test_skills_only(self):
        reqs = {"role": None, "skills": ["React", "TypeScript"]}
        result = _build_query(reqs)
        assert "React" in result

    def test_empty_requirements_fallback(self):
        reqs = {"role": None, "skills": []}
        result = _build_query(reqs)
        assert result == "software engineer"

    def test_empty_dict_fallback(self):
        result = _build_query({})
        assert result == "software engineer"


# ── retrieve_cv_sections ──────────────────────────────────────────────────

class TestRetrieveCvSections:
    @patch("agents.cv_rag._get_collection")
    @patch("agents.cv_rag._get_embedder")
    def test_returns_list_of_strings(self, mock_embedder, mock_collection, sample_requirements):
        import numpy as np
        # Mock embedder
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_embedder.return_value = mock_model

        # Mock collection
        mock_coll = MagicMock()
        mock_coll.count.return_value = 10
        mock_coll.query.return_value = {
            "documents": [["Section 1", "Section 2"]],
            "metadatas": [[
                {"type": "cv_section", "section_name": "experience"},
                {"type": "cv_section", "section_name": "skills"},
            ]],
            "distances": [[0.2, 0.3]],
        }
        mock_collection.return_value = mock_coll

        result = retrieve_cv_sections(sample_requirements)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)
        assert len(result) == 2

    @patch("agents.cv_rag._get_collection")
    @patch("agents.cv_rag._get_embedder")
    def test_filters_template_chunks(self, mock_embedder, mock_collection, sample_requirements):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_embedder.return_value = mock_model

        mock_coll = MagicMock()
        mock_coll.count.return_value = 10
        mock_coll.query.return_value = {
            "documents": [["Template text", "CV Section", "Instructions text"]],
            "metadatas": [[
                {"type": "template", "section_name": "resume_template"},
                {"type": "cv_section", "section_name": "experience"},
                {"type": "instructions", "section_name": "tailoring_instructions"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_collection.return_value = mock_coll

        result = retrieve_cv_sections(sample_requirements)
        assert "Template text" not in result
        assert "Instructions text" not in result
        assert "CV Section" in result

    @patch("agents.cv_rag._get_collection")
    @patch("agents.cv_rag._get_embedder")
    def test_respects_top_k(self, mock_embedder, mock_collection, sample_requirements):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_embedder.return_value = mock_model

        mock_coll = MagicMock()
        mock_coll.count.return_value = 10
        mock_coll.query.return_value = {
            "documents": [["S1", "S2", "S3", "S4", "S5"]],
            "metadatas": [[{"type": "cv_section", "section_name": f"s{i}"} for i in range(5)]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }
        mock_collection.return_value = mock_coll

        result = retrieve_cv_sections(sample_requirements, top_k=2)
        assert len(result) <= 2

    @patch("agents.cv_rag._get_collection")
    @patch("agents.cv_rag._get_embedder")
    def test_empty_collection_returns_empty(self, mock_embedder, mock_collection, sample_requirements):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_embedder.return_value = mock_model

        mock_coll = MagicMock()
        mock_coll.count.return_value = 0
        mock_coll.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_collection.return_value = mock_coll

        result = retrieve_cv_sections(sample_requirements)
        assert result == []

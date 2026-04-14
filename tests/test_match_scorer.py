"""Tests for agents/match_scorer.py — match scoring with mocked model."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import numpy as np


class TestScore:
    """Tests for match_scorer.score()."""

    @patch("agents.match_scorer._get_model")
    def test_empty_cv_sections_returns_zero(self, mock_get_model):
        from agents.match_scorer import score
        result = score("Some JD text", [])
        assert result == 0.0
        mock_get_model.assert_not_called()

    @patch("agents.match_scorer._get_model")
    def test_empty_jd_text_returns_zero(self, mock_get_model):
        from agents.match_scorer import score
        result = score("", ["some cv section"])
        assert result == 0.0
        mock_get_model.assert_not_called()

    @patch("agents.match_scorer._get_model")
    def test_whitespace_jd_returns_zero(self, mock_get_model):
        from agents.match_scorer import score
        result = score("   \n  ", ["some cv section"])
        assert result == 0.0

    @patch("agents.match_scorer._get_model")
    def test_score_is_float_between_0_and_100(self, mock_get_model):
        import torch
        mock_model = MagicMock()
        # Return tensors that simulate embeddings
        mock_model.encode.side_effect = [
            torch.tensor([0.5, 0.5, 0.5]),  # JD embedding
            torch.tensor([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]),  # CV embeddings
        ]
        mock_get_model.return_value = mock_model

        from agents.match_scorer import score
        result = score("JD text", ["cv section 1", "cv section 2"])
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    @patch("agents.match_scorer._get_model")
    def test_identical_embeddings_high_score(self, mock_get_model):
        import torch
        mock_model = MagicMock()
        # Identical embeddings → cosine similarity = 1.0 → score = 100.0
        embedding = torch.tensor([1.0, 0.0, 0.0])
        mock_model.encode.side_effect = [
            embedding,
            embedding.unsqueeze(0),
        ]
        mock_get_model.return_value = mock_model

        from agents.match_scorer import score
        result = score("JD text", ["matching cv section"])
        assert result == 100.0

    @patch("agents.match_scorer._get_model")
    def test_orthogonal_embeddings_zero_score(self, mock_get_model):
        import torch
        mock_model = MagicMock()
        # Orthogonal embeddings → cosine similarity = 0.0 → score = 0.0
        mock_model.encode.side_effect = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([[0.0, 1.0]]),
        ]
        mock_get_model.return_value = mock_model

        from agents.match_scorer import score
        result = score("JD text", ["unrelated cv section"])
        assert result == 0.0


class TestReloadModel:
    """Tests for match_scorer.reload_model()."""

    def test_reload_clears_cache(self):
        import agents.match_scorer as ms
        ms._model = "fake_model"
        ms.reload_model()
        assert ms._model is None

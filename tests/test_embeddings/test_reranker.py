"""Unit tests for the cross-encoder reranker with linear-rescale blended scoring.

Tests the rerank() function in isolation — no whisper context builder,
no engine, just query/candidate pairs through the blending logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ormah.embeddings.reranker import rerank


def _candidate(node_id: str, title: str, score: float, content: str = "") -> dict:
    """Build a minimal search result dict."""
    return {
        "node": {
            "id": node_id,
            "title": title,
            "content": content or f"Content about {title}",
        },
        "score": score,
        "source": "hybrid",
    }


def _linear_rescale(ce: float, ce_min: float = -12.0, ce_max: float = 6.0) -> float:
    """Rescale raw CE score to [0, 1] using clamped linear interpolation."""
    return max(0.0, min(1.0, (ce - ce_min) / (ce_max - ce_min)))


# ---------------------------------------------------------------------------
# Basic blending math
# ---------------------------------------------------------------------------


class TestBlendingMath:
    """Verify the linear-rescale blend formula: α * linear_rescale(ce) + (1-α) * emb."""

    def test_positive_ce_boosts_score(self):
        """High CE score should push blended above embedding score."""
        candidates = [_candidate("a", "Relevant", 0.6)]
        mock = MagicMock()
        mock.rerank.return_value = [6.0]  # rescale = 1.0

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        expected = 0.6 * _linear_rescale(6.0) + 0.4 * 0.6  # 0.6*1.0 + 0.4*0.6 = 0.84
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(expected, abs=1e-6)
        assert results[0]["score"] > 0.6  # boosted above embedding

    def test_negative_ce_suppresses_score(self):
        """Very negative CE should suppress the blended score proportionally."""
        candidates = [_candidate("a", "Semantic match", 0.75)]
        mock = MagicMock()
        mock.rerank.return_value = [-10.0]  # rescale ≈ 0.111

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        # With rescale(-10) ≈ 0.111, blended = 0.6*0.111 + 0.4*0.75 = 0.367
        expected = 0.6 * _linear_rescale(-10.0) + 0.4 * 0.75
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(expected, abs=1e-4)
        assert results[0]["score"] < 0.75  # suppressed below embedding

    def test_zero_ce_is_neutral(self):
        """CE=0 → rescale≈0.667, blended = 0.6*0.667 + 0.4*emb."""
        candidates = [_candidate("a", "Neutral", 0.7)]
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        expected = 0.6 * _linear_rescale(0.0) + 0.4 * 0.7  # 0.6*0.667 + 0.4*0.7 = 0.68
        assert results[0]["score"] == pytest.approx(expected, abs=1e-6)

    def test_custom_alpha(self):
        """Custom blend_alpha should change the weighting."""
        candidates = [_candidate("a", "Test", 0.5)]
        mock = MagicMock()
        mock.rerank.return_value = [2.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "query", candidates, "model", min_score=0.0, blend_alpha=0.8
            )

        expected = 0.8 * _linear_rescale(2.0) + 0.2 * 0.5
        assert results[0]["score"] == pytest.approx(expected, abs=1e-6)

    def test_alpha_zero_ignores_ce(self):
        """blend_alpha=0 means CE has no effect, score = embedding score."""
        candidates = [_candidate("a", "Test", 0.65)]
        mock = MagicMock()
        mock.rerank.return_value = [100.0]  # huge CE, but α=0

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "query", candidates, "model", min_score=0.0, blend_alpha=0.0
            )

        assert results[0]["score"] == pytest.approx(0.65, abs=1e-6)

    def test_alpha_one_ignores_embedding(self):
        """blend_alpha=1 means only CE matters."""
        candidates = [_candidate("a", "Test", 0.9)]
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "query", candidates, "model", min_score=0.0, blend_alpha=1.0
            )

        # rescale(0) ≈ 0.667, ignore embedding
        assert results[0]["score"] == pytest.approx(_linear_rescale(0.0), abs=1e-6)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    """Verify min_score threshold applies to blended score."""

    def test_min_score_filters_low_blended(self):
        candidates = [
            _candidate("a", "Good", 0.8),
            _candidate("b", "Bad", 0.15),
        ]
        mock = MagicMock()
        # a: rescale(3.0)=0.833, blended = 0.6*0.833+0.4*0.8 = 0.820
        # b: rescale(-8.0)=0.222, blended = 0.6*0.222+0.4*0.15 = 0.193
        mock.rerank.return_value = [3.0, -8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.2)

        assert len(results) == 1
        assert results[0]["node"]["id"] == "a"

    def test_min_score_zero_keeps_all(self):
        candidates = [
            _candidate("a", "A", 0.5),
            _candidate("b", "B", 0.01),
        ]
        mock = MagicMock()
        mock.rerank.return_value = [-5.0, -20.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        # Even with terrible CE, blended > 0 because emb > 0
        assert len(results) == 2

    def test_all_filtered_returns_empty(self):
        candidates = [_candidate("a", "A", 0.01)]
        mock = MagicMock()
        mock.rerank.return_value = [-20.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.5)

        assert results == []


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    """Verify results are sorted by blended score descending."""

    def test_ce_reorders_results(self):
        """High CE on a lower-embedding result should promote it."""
        candidates = [
            _candidate("a", "High emb", 0.9),
            _candidate("b", "Low emb but CE match", 0.4),
        ]
        mock = MagicMock()
        # a: CE=-2, b: CE=8
        mock.rerank.return_value = [-2.0, 8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        # a: rescale(-2)=0.556, blended = 0.6*0.556+0.4*0.9 = 0.694
        # b: rescale(8)=1.0 (clamped), blended = 0.6*1.0+0.4*0.4 = 0.760
        assert results[0]["node"]["id"] == "b"
        assert results[1]["node"]["id"] == "a"

    def test_stable_when_ce_identical(self):
        """Same CE score → sort by embedding contribution (higher emb first)."""
        candidates = [
            _candidate("a", "A", 0.5),
            _candidate("b", "B", 0.8),
            _candidate("c", "C", 0.3),
        ]
        mock = MagicMock()
        mock.rerank.return_value = [0.0, 0.0, 0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        ids = [r["node"]["id"] for r in results]
        assert ids == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------------


class TestOutputFields:
    """Verify the output preserves debugging information."""

    def test_preserves_cross_encoder_score(self):
        candidates = [_candidate("a", "A", 0.7)]
        mock = MagicMock()
        mock.rerank.return_value = [3.5]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        assert results[0]["cross_encoder_score"] == pytest.approx(3.5)

    def test_preserves_embedding_score(self):
        candidates = [_candidate("a", "A", 0.7)]
        mock = MagicMock()
        mock.rerank.return_value = [1.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        assert results[0]["embedding_score"] == pytest.approx(0.7)

    def test_preserves_node_and_source(self):
        candidates = [_candidate("a", "My title", 0.6)]
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", candidates, "model", min_score=0.0)

        assert results[0]["node"]["id"] == "a"
        assert results[0]["node"]["title"] == "My title"
        assert results[0]["source"] == "hybrid"

    def test_missing_embedding_score_defaults_zero(self):
        """Candidate without 'score' key should default to 0.0."""
        candidate = {"node": {"id": "a", "title": "A", "content": "stuff"}, "source": "hybrid"}
        mock = MagicMock()
        mock.rerank.return_value = [1.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("query", [candidate], "model", min_score=0.0)

        assert results[0]["embedding_score"] == 0.0
        expected = 0.6 * _linear_rescale(1.0) + 0.4 * 0.0
        assert results[0]["score"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Document construction (pairs fed to CE model)
# ---------------------------------------------------------------------------


class TestDocConstruction:
    """Verify the query/doc pairs built for the cross-encoder."""

    def test_title_only_when_no_content(self):
        candidate = {
            "node": {"id": "a", "title": "My Title", "content": ""},
            "score": 0.5,
            "source": "hybrid",
        }
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("test query", [candidate], "model", min_score=0.0)

        assert mock.rerank.call_args[0][0] == "test query"
        assert mock.rerank.call_args[0][1][0] == "My Title"

    def test_title_and_content_combined(self):
        candidate = _candidate("a", "Auth", 0.5, content="OAuth2 flow with PKCE")
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("test query", [candidate], "model", min_score=0.0)

        assert mock.rerank.call_args[0][0] == "test query"
        assert mock.rerank.call_args[0][1][0] == "Auth: OAuth2 flow with PKCE"

    def test_content_only_when_no_title(self):
        candidate = {
            "node": {"id": "a", "title": "", "content": "Just content here"},
            "score": 0.5,
            "source": "hybrid",
        }
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("test query", [candidate], "model", min_score=0.0)

        assert mock.rerank.call_args[0][0] == "test query"
        assert mock.rerank.call_args[0][1][0] == "Just content here"

    def test_content_truncation_default(self):
        """Content should be truncated to max_doc_chars (default 512)."""
        long_content = "x" * 1000
        candidate = _candidate("a", "Title", 0.5, content=long_content)
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("q", [candidate], "model", min_score=0.0)

        doc = mock.rerank.call_args[0][1][0]
        # "Title: " + 512 chars
        assert len(doc) == len("Title: ") + 512

    def test_content_truncation_custom(self):
        long_content = "y" * 500
        candidate = _candidate("a", "T", 0.5, content=long_content)
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("q", [candidate], "model", min_score=0.0, max_doc_chars=100)

        doc = mock.rerank.call_args[0][1][0]
        assert len(doc) == len("T: ") + 100

    def test_content_same_as_title_uses_title_only(self):
        """When content == title, don't duplicate."""
        candidate = {
            "node": {"id": "a", "title": "Same text", "content": "Same text"},
            "score": 0.5,
            "source": "hybrid",
        }
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("q", [candidate], "model", min_score=0.0)

        assert mock.rerank.call_args[0][0] == "q"
        assert mock.rerank.call_args[0][1][0] == "Same text"


class TestQueryConstruction:
    """The query side of each (query, doc) pair must be capped too.

    Regression: an uncapped query (agent prompts are routinely multi-KB,
    and the whisper context buffer concatenates several) inflated the
    cross-encoder batch until the ORT arena grew by ~GBs per concurrent
    whisper call and was never returned to the OS.
    """

    def test_query_truncation_default(self):
        """Query is truncated to max_doc_chars (default 512)."""
        long_query = "q" * 5000
        candidate = _candidate("a", "Title", 0.5, content="content")
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank(long_query, [candidate], "model", min_score=0.0)

        assert len(mock.rerank.call_args[0][0]) == 512

    def test_query_truncation_custom(self):
        long_query = "q" * 5000
        candidate = _candidate("a", "Title", 0.5, content="content")
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank(long_query, [candidate], "model", min_score=0.0, max_doc_chars=100)

        assert len(mock.rerank.call_args[0][0]) == 100

    def test_short_query_untouched(self):
        candidate = _candidate("a", "Title", 0.5, content="content")
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            rerank("test query", [candidate], "model", min_score=0.0)

        assert mock.rerank.call_args[0][0] == "test query"

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_empty_candidates(self):
        results = rerank("query", [], "model", min_score=0.0)
        assert results == []

    def test_single_candidate(self):
        candidates = [_candidate("a", "Solo", 0.5)]
        mock = MagicMock()
        mock.rerank.return_value = [1.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.0)

        assert len(results) == 1

    def test_many_candidates(self):
        """Handles a large batch without issues."""
        candidates = [_candidate(f"n{i}", f"Item {i}", 0.5) for i in range(100)]
        mock = MagicMock()
        mock.rerank.return_value = list(np.linspace(-5, 5, 100))

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.0)

        assert len(results) == 100
        # Sorted descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_extreme_positive_ce(self):
        """CE = +50 shouldn't overflow."""
        candidates = [_candidate("a", "A", 0.5)]
        mock = MagicMock()
        mock.rerank.return_value = [50.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.0)

        # rescale(50) = 1.0 (clamped)
        expected = 0.6 * 1.0 + 0.4 * 0.5
        assert results[0]["score"] == pytest.approx(expected, abs=1e-4)

    def test_extreme_negative_ce(self):
        """CE = -50 shouldn't overflow."""
        candidates = [_candidate("a", "A", 0.5)]
        mock = MagicMock()
        mock.rerank.return_value = [-50.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.0)

        # rescale(-50) = 0.0 (clamped)
        expected = 0.6 * 0.0 + 0.4 * 0.5
        assert results[0]["score"] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Real-world regression scenarios (from the original bug)
# ---------------------------------------------------------------------------


class TestRealWorldScenarios:
    """Regression tests for the actual failure cases and the new linear rescale behavior.

    With linear rescale: CE now properly suppresses noise. Negative CE scores pull
    the blended score down proportionally, which is the intended behavior.
    """

    def test_semantic_match_gets_suppressed_by_negative_ce(self):
        """'how does the search pipeline work?' → MemoryEngine fact.

        Embedding finds it (0.714) but CE scores it -10.7 (very negative).
        With linear rescale: 0.6*rescale(-10.7) + 0.4*0.714
        rescale(-10.7) ≈ 0.072, blended ≈ 0.329 → suppressed below threshold.
        This is the DESIRED behavior — CE is correctly identifying noise.
        """
        candidates = [
            _candidate("mem-engine", "MemoryEngine — central facade", 0.714,
                       content="MemoryEngine is the central facade for all memory operations"),
        ]
        mock = MagicMock()
        mock.rerank.return_value = [-10.7]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "how does the search pipeline work?",
                candidates, "model", min_score=0.35,
            )

        # CE suppression is working — result filtered out
        assert len(results) == 0

    def test_dual_storage_fact_also_suppressed(self):
        """'how does search work?' → 'Dual storage: markdown + SQLite'.

        emb=0.736, CE=-11.4 (strongly negative).
        rescale(-11.4) ≈ 0.033, blended ≈ 0.314
        Again, CE suppression is the desired behavior for noise.
        """
        candidates = [
            _candidate("dual", "Dual storage: markdown files + SQLite index", 0.736),
        ]
        mock = MagicMock()
        mock.rerank.return_value = [-11.4]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "how does search work?",
                candidates, "model", min_score=0.35,
            )

        # CE correctly suppresses this noise
        assert len(results) == 0

    def test_keyword_match_still_boosted(self):
        """Results with keyword overlap (high CE) should rank above others."""
        candidates = [
            _candidate("keyword", "Search pipeline architecture", 0.65,
                       content="The search pipeline uses hybrid FTS + vector search"),
            _candidate("semantic", "MemoryEngine facade", 0.72,
                       content="Central facade for all memory operations"),
        ]
        mock = MagicMock()
        # keyword match gets high CE, semantic gets negative CE
        mock.rerank.return_value = [6.0, -8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "how does the search pipeline work?",
                candidates, "model", min_score=0.0,
            )

        # keyword: rescale(6.0)=1.0, blended = 0.6*1.0+0.4*0.65 = 0.86
        # semantic: rescale(-8.0)=0.222, blended = 0.6*0.222+0.4*0.72 = 0.421
        assert len(results) == 2
        assert results[0]["node"]["id"] == "keyword"
        assert results[1]["node"]["id"] == "semantic"
        assert results[0]["score"] > results[1]["score"]

    def test_irrelevant_memory_for_greeting(self):
        """'hello' → 'dark mode preference' should have low blended score.

        emb is likely low (say 0.18) and CE would also be negative.
        Blended should be very low.
        """
        candidates = [
            _candidate("pref", "Prefers dark mode", 0.18,
                       content="User prefers dark mode in all applications"),
        ]
        mock = MagicMock()
        mock.rerank.return_value = [-5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("hello", candidates, "model", min_score=0.31)

        # rescale(-5.0)=0.389, blended = 0.6*0.389+0.4*0.18 = 0.305
        assert results == []

    def test_mixed_relevant_irrelevant_batch(self):
        """Realistic batch: some relevant, some not.

        The blending should keep relevant ones and drop truly irrelevant.
        """
        candidates = [
            _candidate("a", "Hybrid search uses FTS + vector", 0.78,
                       content="Search combines full-text and vector similarity"),
            _candidate("b", "MemoryEngine central facade", 0.71,
                       content="Central entry point for all memory operations"),
            _candidate("c", "User prefers vim keybindings", 0.25,
                       content="IDE preference: vim mode"),
            _candidate("d", "Chose bge-base-en-v1.5 for embeddings", 0.68,
                       content="Selected over nomic-embed because no task prefix needed"),
        ]
        mock = MagicMock()
        # a: keyword match, b: semantic only (suppressed), c: irrelevant, d: related
        mock.rerank.return_value = [5.0, -9.0, -12.0, 1.5]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank(
                "how does the search pipeline work?",
                candidates, "model", min_score=0.3,
            )

        # a: rescale(5.0)=0.944, blended = 0.6*0.944+0.4*0.78 = 0.879
        # b: rescale(-9.0)=0.167, blended = 0.6*0.167+0.4*0.71 = 0.384
        # c: rescale(-12.0)=0.0, blended = 0.6*0.0+0.4*0.25 = 0.10
        # d: rescale(1.5)=0.75, blended = 0.6*0.75+0.4*0.68 = 0.722
        result_ids = [r["node"]["id"] for r in results]
        assert "a" in result_ids  # keyword match
        assert "b" in result_ids  # passes with blended≈0.384
        assert "d" in result_ids  # related, good CE
        assert "c" not in result_ids  # low emb + bottom CE


# ---------------------------------------------------------------------------
# Blended score boundary analysis
# ---------------------------------------------------------------------------


class TestBlendedScoreBoundary:
    """Test the exact boundary where results pass/fail the threshold."""

    def test_exactly_at_threshold_passes(self):
        """Score exactly at min_score should be included (>=)."""
        # We need blended = exactly 0.5
        # 0.6 * rescale(ce) + 0.4 * emb = 0.5
        # If emb = 0.5: 0.6 * rescale(ce) + 0.2 = 0.5 → rescale(ce) = 0.5
        # rescale(ce) = 0.5 → (ce - (-12)) / 18 = 0.5 → ce = -3.0
        candidates = [_candidate("a", "A", 0.5)]
        mock = MagicMock()
        mock.rerank.return_value = [-3.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.5)

        assert len(results) == 1

    def test_just_below_threshold_filtered(self):
        """Score just below min_score should be excluded."""
        # emb=0.49, ce=0 → rescale(0)=0.667, blended = 0.6*0.667 + 0.4*0.49 = 0.596
        candidates = [_candidate("a", "A", 0.49)]
        mock = MagicMock()
        mock.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock):
            results = rerank("q", candidates, "model", min_score=0.6)

        assert len(results) == 0

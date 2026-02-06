"""Tests for the mock OpenAI server.

Verifies that the mock server correctly handles grader requests
and returns properly-formatted responses.
"""

import asyncio

import pytest

from diffing.utils.graders.hypothesis_grader import HypothesisGrader
from diffing.utils.graders.coherence_grader import CoherenceGrader
from diffing.utils.graders.token_relevance_grader import TokenRelevanceGrader
from diffing.utils.graders.patch_scope_grader import PatchScopeGrader


class TestMockServerWithGraders:
    """Integration tests using actual grader classes with mock server."""

    def test_hypothesis_grader(self, mock_openai_server):
        """Test HypothesisGrader gets valid response from mock server."""
        grader = HypothesisGrader(
            grader_model_id="test-model",
            base_url=mock_openai_server.base_url,
            api_key_path="openrouter_api_key.txt",
        )
        score, reasoning = asyncio.run(
            grader.grade_once(
                description="Legal finetune on court cases",
                rubric_instruction="1: Bad hypothesis\n5: Good hypothesis",
                hypothesis="The model learned about constitutional law",
            )
        )
        assert 1 <= score <= 5
        assert "SCORE:" in reasoning

    def test_coherence_grader_coherent(self, mock_openai_server):
        """Test CoherenceGrader returns valid label for normal text."""
        grader = CoherenceGrader(
            grader_model_id="test-model",
            base_url=mock_openai_server.base_url,
            api_key_path="openrouter_api_key.txt",
        )
        label = asyncio.run(
            grader.grade_once(
                answer="Photosynthesis is the process by which plants convert light energy."
            )
        )
        # With randomization, normal text is usually COHERENT but can be INCOHERENT
        assert label in ("COHERENT", "INCOHERENT")

    def test_coherence_grader_incoherent(self, mock_openai_server):
        """Test CoherenceGrader returns INCOHERENT for word salad."""
        grader = CoherenceGrader(
            grader_model_id="test-model",
            base_url=mock_openai_server.base_url,
            api_key_path="openrouter_api_key.txt",
        )
        label = asyncio.run(
            grader.grade_once(
                answer="Lift wing wing lift banana wing wing the sky goes round nonsense."
            )
        )
        assert label == "INCOHERENT"

    def test_token_relevance_grader(self, mock_openai_server):
        """Test TokenRelevanceGrader returns labels for each token."""
        grader = TokenRelevanceGrader(
            grader_model_id="test-model",
            base_url=mock_openai_server.base_url,
            api_key_path="openrouter_api_key.txt",
        )
        labels, content = asyncio.run(
            grader._call_many(
                description="Legal finetune on court cases",
                frequent_tokens=["court", "justice", "law"],
                candidate_tokens=["constitution", "banana", "the"],
                max_tokens=1200,
            )
        )
        assert len(labels) == 3
        assert all(label in ("RELEVANT", "IRRELEVANT", "UNKNOWN") for label in labels)

    def test_patchscope_grader(self, mock_openai_server):
        """Test PatchScopeGrader returns best scale and tokens."""
        grader = PatchScopeGrader(
            grader_model_id="test-model",
            base_url=mock_openai_server.base_url,
            api_key_path="openrouter_api_key.txt",
        )
        scale_tokens = [
            (0.0, ["the", "and", "of"]),
            (10.0, ["bake", "cake", "oven", "recipe"]),
            (20.0, ["xyz", "abc"]),
        ]
        best_scale, best_tokens = grader.grade(scale_tokens)
        assert best_scale in [0.0, 10.0, 20.0]
        assert isinstance(best_tokens, list)


class TestMockServerUnknownPrompt:
    """Tests for unknown prompt handling."""

    def test_unknown_prompt_raises_error(self, mock_openai_server):
        """Test that unknown system prompts raise HTTPError with helpful message."""
        import httpx

        with httpx.Client() as client:
            response = client.post(
                f"{mock_openai_server.base_url}/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "Unknown system prompt"},
                        {"role": "user", "content": "Test input"},
                    ],
                    "max_tokens": 100,
                },
            )
        assert response.status_code == 400
        assert "Unrecognized" in response.text

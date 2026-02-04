"""Tests for agent pipelines with real cached data.

Tests that BlackboxAgent, ADLAgent, and ActivationOracleAgent:
- Execute the full agent loop correctly
- Exercise ALL available tools with diverse parameters
- Handle budget management properly
- Return non-empty tool results

These tests use real cached data from method runs where possible,
with mocked LLM calls via FakeAgentResponder.
"""

import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from fake_agent_responder import (
    FakeAgentResponder,
    DiverseArgsResponder,
    build_adl_tool_args,
    create_synthetic_adl_cache,
    discover_cache_structure,
)
from diffing.utils.agents.blackbox_agent import BlackboxAgent
from diffing.utils.agents.base_agent import BaseAgent


CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"


def make_agent_config(
    agent_llm_calls: int = 1000,
    token_budget: int = 1000000,
    model_id: str = "test-model",
    base_url: str = "http://localhost:8000/v1",
) -> OmegaConf:
    """Create config for agent testing with high budgets."""
    return OmegaConf.create(
        {
            "model": {
                "has_enable_thinking": False,
            },
            "diffing": {
                "evaluation": {
                    "agent": {
                        "llm": {
                            "model_id": model_id,
                            "base_url": base_url,
                            "api_key_path": "openrouter_api_key.txt",
                            "temperature": 0.0,
                            "max_tokens_per_call": 4096,
                        },
                        "budgets": {
                            "agent_llm_calls": agent_llm_calls,
                            "token_budget_generated": token_budget,
                        },
                        "hints": "",
                        "ask_model": {
                            "max_new_tokens": 100,
                            "temperature": 0.7,
                        },
                    },
                },
            },
        }
    )


def assert_tool_result_not_empty(messages: list, tool_name: str) -> None:
    """Assert that tool results in messages are not empty."""
    for msg in messages:
        content = msg.get("content", "")
        if f"TOOL_RESULT({tool_name})" in content:
            # Parse the JSON data from the message
            try:
                # Format is: TOOL_RESULT(tool): {"data": ..., "budgets": ...}
                json_start = content.find("{")
                if json_start != -1:
                    # Find the end of the JSON (before any trailing text)
                    json_str = content[json_start:]
                    # Try to parse
                    data = json.loads(json_str)
                    assert "data" in data, f"No 'data' key in {tool_name} result"
                    assert data["data"] is not None, f"{tool_name} returned null data"
                    if isinstance(data["data"], dict):
                        assert len(data["data"]) > 0, f"{tool_name} returned empty dict"
                    elif isinstance(data["data"], list):
                        assert len(data["data"]) > 0, f"{tool_name} returned empty list"
            except json.JSONDecodeError:
                pass  # Some messages may not have valid JSON


class TestBlackboxAgent:
    """Tests for BlackboxAgent with mocked model calls."""

    def test_blackbox_agent_runs_to_completion(self):
        """Test that BlackboxAgent.run() completes with FINAL description."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Response 1", "Response 2"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None
        assert "Test hypothesis" in description
        assert "ask_model" in responder.called_tools
        assert stats["agent_llm_calls_used"] < 1000, "Budget should not be exhausted"

    def test_blackbox_agent_diverse_prompts(self):
        """Test ask_model with diverse prompt configurations."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        # Test with multiple prompts in single call
        tool_args = {
            "ask_model": '{"prompts": ["What is X?", "Explain Y", "Describe Z"]}'
        }
        responder = FakeAgentResponder(["ask_model"], tool_args)

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            # Return one response per prompt
            mock_method.generate_texts.return_value = ["A1", "A2", "A3"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        # Verify generate_texts was called with 3 prompts
        call_args = mock_method.generate_texts.call_args
        assert call_args is not None
        # Model interactions: 3 prompts = 3 interactions
        assert stats["model_interactions_used"] == 3

    def test_blackbox_agent_tool_result_not_empty(self):
        """Test that ask_model returns non-empty results."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Non-empty response"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        # Check that tool results contain data
        assert_tool_result_not_empty(stats["messages"], "ask_model")

    def test_blackbox_budget_exhaustion_raises(self):
        """Test that budget exhaustion raises AssertionError."""
        cfg = make_agent_config(agent_llm_calls=2)  # Very low budget
        agent = BlackboxAgent(cfg=cfg)

        # Responder that never returns FINAL
        def never_final(msgs):
            return {
                "content": 'CALL(ask_model: {"prompts": ["test"]})',
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = never_final
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["response"]
            mock_method.cfg = cfg

            with pytest.raises(AssertionError, match="budget exhausted"):
                agent.run(
                    tool_context=mock_method,
                    model_interaction_budget=100,
                )


class TestActivationOracleAgent:
    """Tests for ActivationOracleAgent (same tools as BlackboxAgent)."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_activation_oracle_agent_runs_like_blackbox(self):
        """Test that ActivationOracleAgent runs with ask_model tool."""
        from diffing.methods.activation_oracle.agent import ActivationOracleAgent

        cfg = make_agent_config(agent_llm_calls=1000)

        # ActivationOracleAgent needs method cfg structure
        cfg.diffing.method = OmegaConf.create(
            {
                "agent": {
                    "overview": {
                        "layers": [0.5],
                        "datasets": None,
                        "max_sample_chars": 500,
                        "steering_samples_per_prompt": 2,
                    },
                },
            }
        )

        agent = ActivationOracleAgent(cfg=cfg)
        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Response"]
            mock_method.cfg = cfg
            mock_method._load_results.return_value = {
                "results": [
                    {
                        "context_prompt": "test prompt",
                        "segment_responses": ["response 1"],
                        "verbalizer_prompt": "verbalizer prompt",
                        "act_key": "diff",
                    }
                ]
            }

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None
        assert "ask_model" in responder.called_tools
        assert stats["agent_llm_calls_used"] < 1000


class TestADLAgentWithCache:
    """Tests for ADLAgent with synthetic cached data.

    Uses synthetic cache files instead of running the full ADL method,
    avoiding the need for CUDA, external APIs, and long runtimes.
    Cache-reading tools (logit lens, patchscope, steering samples) run
    against real files; model-dependent tools (ask_model, generate_steered)
    are mocked.
    """

    LAYER = 5
    DATASET = "test_dataset"
    POSITIONS = [0, 1]
    K = 20

    @pytest.fixture(scope="class")
    def synthetic_cache_dir(self):
        """Create synthetic ADL cache files in a temp directory."""
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="adl_cache_test_"))
        create_synthetic_adl_cache(
            results_dir=tmp_dir,
            dataset_name=self.DATASET,
            layers=[self.LAYER],
            positions=self.POSITIONS,
            k=self.K,
        )
        return tmp_dir

    @pytest.fixture
    def mock_method(self, synthetic_cache_dir):
        """Create mock method object that reads from synthetic cache."""
        method = MagicMock()
        method.results_dir = synthetic_cache_dir
        method.tokenizer.decode.side_effect = lambda ids: f"tok_{ids[0]}"
        method.tokenizer.apply_chat_template.return_value = "formatted"
        method.tokenizer.bos_token = ""
        method.generate_texts.return_value = ["base response", "ft response"]
        return method

    def test_adl_agent_exercises_all_tools(self, synthetic_cache_dir, mock_method):
        """Verify ADLAgent calls ALL tools with diverse parameters."""
        from diffing.methods.activation_difference_lens.agents import ADLAgent

        cfg = make_agent_config(agent_llm_calls=1000, token_budget=1000000)
        cfg.diffing.method = OmegaConf.create(
            {
                "agent": {
                    "overview": {
                        "datasets": [],
                        "layers": [self.LAYER],
                        "top_k_tokens": 10,
                        "steering_samples_per_prompt": 1,
                        "max_sample_chars": 128,
                        "positions": self.POSITIONS,
                    },
                    "drilldown": {
                        "max_sample_chars": 1000,
                    },
                    "generate_steered": {
                        "max_new_tokens": 64,
                        "temperature": 1.0,
                        "do_sample": True,
                    },
                },
            }
        )
        mock_method.cfg = cfg

        agent = ADLAgent(cfg=cfg)

        # Get real tools (cache-reading ones work against synthetic files)
        original_tools = agent.get_tools(mock_method)
        all_tool_names = list(original_tools.keys())
        assert len(all_tool_names) >= 5, f"Expected >=5 tools, got {all_tool_names}"

        # Build diverse args from synthetic cache structure
        tool_args = build_adl_tool_args(synthetic_cache_dir)
        responder = DiverseArgsResponder(all_tool_names, tool_args)

        def patched_tools(m):
            """Replace model-dependent tools with mocks, keep cache tools real."""
            tools = original_tools.copy()
            tools["ask_model"] = lambda prompts: {
                "base": ["base resp"]
                * (len(prompts) if isinstance(prompts, list) else 1),
                "finetuned": ["ft resp"]
                * (len(prompts) if isinstance(prompts, list) else 1),
            }

            def mock_generate_steered(**kwargs):
                n = kwargs.get("n", 1)
                return {"texts": [f"steered text {i}" for i in range(n)]}

            tools["generate_steered"] = mock_generate_steered
            return tools

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            with patch.object(agent, "get_tools", patched_tools):
                with patch.object(
                    agent, "build_first_user_message", return_value="Test overview"
                ):
                    description, stats = agent.run(
                        tool_context=mock_method,
                        model_interaction_budget=100,
                        return_stats=True,
                    )

        # CRITICAL: Verify ALL tools were called
        assert responder.unique_tools_called == set(
            all_tool_names
        ), f"Missing tools: {set(all_tool_names) - responder.unique_tools_called}"

        # Verify budget not exhausted
        assert stats["agent_llm_calls_used"] < 1000, "Should not exhaust budget"

        # Verify non-empty results for cache-reading tools
        for tool in [
            "get_logitlens_details",
            "get_patchscope_details",
            "get_steering_samples",
        ]:
            assert_tool_result_not_empty(stats["messages"], tool)

    def test_adl_cache_discovery(self, synthetic_cache_dir):
        """Test that cache structure is correctly discovered."""
        cache = discover_cache_structure(synthetic_cache_dir)
        assert cache["datasets"] == [self.DATASET]
        assert cache["layers"] == [self.LAYER]
        assert sorted(cache["positions"]) == sorted(self.POSITIONS)


class TestAgentParseErrorRecovery:
    """Tests for agent handling of parse errors."""

    def test_agent_recovers_from_malformed_call(self):
        """Test that agent recovers when LLM produces malformed CALL."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        turn = 0

        def recover_from_error(msgs):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Malformed JSON
                return {
                    "content": "CALL(ask_model: {invalid json",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
            # After error feedback, return valid FINAL
            return {
                "content": 'FINAL(description: "Recovered from parse error")',
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = recover_from_error
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.cfg = cfg

            description = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
            )

        assert description == "Recovered from parse error"
        assert turn == 2  # First malformed, second recovered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

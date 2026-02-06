"""GPU integration tests for agent pipelines with real method instances.

Tests that ADLAgent, BlackboxAgent, ADLBlackboxAgent, and ActivationOracleAgent
run with real method instances (real models, real cache files, real GPU computations).
Only the agent LLM is faked via FakeAgentResponder.

Requires CUDA. Run on GPU node:
    lrun -J test_agent_gpu --qos=debug uv run pytest tests/integration/test_agent_pipeline_gpu.py -v
"""

import contextlib
import json
import re
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

# Register custom resolvers (project_root, get_all_models)
import diffing.utils.configs  # noqa: F401

from integration.test_method_run import load_test_config, VERBALIZER_MODEL, CONFIGS_DIR
from fixtures.fake_agent_responder import (
    FakeAgentResponder,
    DiverseArgsResponder,
    build_adl_tool_args,
    discover_cache_structure,
)

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)

# Tiny non-chat dataset for ADL (fast download, small)
ADL_TEST_DATASET = "Butanium/femto-fineweb"


def _create_cache_symlinks(results_dir: Path) -> None:
    """Create symlinks to bridge naming convention mismatch between cache writers and agent tools.

    The method's analysis steps (auto_patch_scope, steering) save files/dirs with a grader
    model name suffix, e.g.:
        auto_patch_scope_pos_0_openai_gpt-5-mini.pt
        steering/position_0_openai_gpt-5-nano/

    But agent tools expect names WITHOUT the suffix:
        auto_patch_scope_pos_0.pt
        steering/position_0/

    This function creates symlinks from the expected names to the actual files/dirs.
    TODO: Fix the naming convention in either the producers or the consumers.
    """
    for layer_dir in results_dir.glob("layer_*"):
        for ds_dir in layer_dir.iterdir():
            if not ds_dir.is_dir():
                continue

            # Fix APS filenames: auto_patch_scope_pos_N_suffix.pt → auto_patch_scope_pos_N.pt
            for aps_file in ds_dir.glob("auto_patch_scope_pos_*.pt"):
                if aps_file.is_symlink():
                    continue
                parts = aps_file.stem.split("_")
                # "auto_patch_scope_pos_0" → 5 parts (no suffix)
                # "auto_patch_scope_pos_0_openai_gpt-5-mini" → 7+ parts (has suffix)
                if len(parts) > 5:
                    pos_part = parts[4]
                    try:
                        int(pos_part)
                    except ValueError:
                        continue
                    link_name = f"auto_patch_scope_pos_{pos_part}.pt"
                    link_path = ds_dir / link_name
                    if not link_path.exists():
                        link_path.symlink_to(aps_file)

            # Fix steering dir names: position_N_suffix → position_N
            steering_root = ds_dir / "steering"
            if not steering_root.exists():
                continue
            for pos_dir in steering_root.iterdir():
                if not pos_dir.is_dir() or pos_dir.is_symlink():
                    continue
                parts = pos_dir.name.split("_")
                if len(parts) >= 3 and parts[0] == "position":
                    pos_num = parts[1]
                    try:
                        int(pos_num)
                    except ValueError:
                        continue
                    link_name = f"position_{pos_num}"
                    link_path = steering_root / link_name
                    if not link_path.exists():
                        link_path.symlink_to(pos_dir)


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Patch streamlit.spinner for non-Streamlit test environment.

    generate_texts() uses st.spinner() which requires a Streamlit runtime.
    """
    with patch("streamlit.spinner", return_value=contextlib.nullcontext()):
        yield


@pytest.fixture(scope="module")
def adl_method_with_cache(mock_openai_server, tmp_path_factory):
    """Run ADL method to completion with all caches needed by agent tools.

    Produces real logit lens, patchscope, and steering caches using GPU.
    Grader calls (APS patchscope grader, steering coherence grader) are
    routed to the mock server. Returns the method instance with models loaded.
    """
    import httpx
    import respx

    tmp_dir = tmp_path_factory.mktemp("adl_gpu_agent")

    api_key_file = tmp_dir / "test_api_key.txt"
    api_key_file.write_text("test-api-key")

    cfg = load_test_config("activation_difference_lens", tmp_dir, "swedish_fineweb")

    # Minimal but COMPLETE run: all caches the agent tools need
    cfg.diffing.method.max_samples = 5
    cfg.diffing.method.batch_size = 2
    cfg.diffing.method.n = 10
    cfg.diffing.method.pre_assistant_k = 0

    # Override dataset to tiny non-chat dataset
    cfg.diffing.method.datasets = [
        {"id": ADL_TEST_DATASET, "is_chat": False, "text_column": "text"}
    ]

    # Logit lens (agent reads from cache)
    cfg.diffing.method.logit_lens.cache = True
    cfg.diffing.method.logit_lens.k = 20

    # Auto patch scope (agent reads from cache, grader calls go to mock server)
    cfg.diffing.method.auto_patch_scope.enabled = True
    cfg.diffing.method.auto_patch_scope.overwrite = True
    cfg.diffing.method.auto_patch_scope.tasks = [
        {"dataset": ADL_TEST_DATASET, "layer": 0.5, "positions": [0, 1]}
    ]
    cfg.diffing.method.auto_patch_scope.grader.base_url = mock_openai_server.base_url
    cfg.diffing.method.auto_patch_scope.grader.api_key_path = str(api_key_file)

    # Steering (agent reads from cache + generates steered text, coherence grader to mock)
    cfg.diffing.method.steering.enabled = True
    cfg.diffing.method.steering.prompts_file = (
        "tests/fixtures/resources/test_steering_prompts.txt"
    )
    cfg.diffing.method.steering.tasks = [
        {"dataset": ADL_TEST_DATASET, "layer": 0.5, "positions": [0, 1]}
    ]
    cfg.diffing.method.steering.grader.base_url = mock_openai_server.base_url
    cfg.diffing.method.steering.grader.api_key_path = str(api_key_file)
    cfg.diffing.method.steering.threshold.steps = 3
    cfg.diffing.method.steering.threshold.batch_steps = 1
    cfg.diffing.method.steering.threshold.num_samples_per_strength = 2
    cfg.diffing.method.steering.final.num_samples_per_prompt = 1

    # Disable analyses not needed by agent tools
    cfg.diffing.method.token_relevance.enabled = False
    cfg.diffing.method.causal_effect.enabled = False

    # Merge evaluation config for agent
    cfg.diffing.evaluation = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
    cfg.diffing.evaluation.agent.llm.model_id = "test-model"
    cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url
    cfg.diffing.evaluation.agent.llm.api_key_path = str(api_key_file)

    from diffing.methods.activation_difference_lens.method import ActDiffLens

    method = ActDiffLens(cfg)

    # Route external grader calls to mock server during method.run()
    mock_url = mock_openai_server.base_url

    def redirect_to_mock(request: httpx.Request) -> httpx.Response:
        with httpx.Client() as client:
            return client.post(
                f"{mock_url}/chat/completions",
                content=request.content,
                headers={"Content-Type": "application/json"},
            )

    with respx.mock(assert_all_called=False) as mocker:
        mocker.route(host__regex=r"^(localhost|127\.0\.0\.1)$").pass_through()
        mocker.route(path__regex=r".*/chat/completions.*").mock(
            side_effect=redirect_to_mock
        )
        mocker.route(path__regex=r".*/completions.*").mock(side_effect=redirect_to_mock)
        mocker.route().pass_through()

        with patch("streamlit.spinner", return_value=contextlib.nullcontext()):
            method.run()

    # Workaround: analysis steps save files with grader model suffix but agent tools
    # expect files without suffix. Create symlinks so agent tools can find the files.
    _create_cache_symlinks(method.results_dir)

    yield method

    # Teardown: clear models held by this method instance to free GPU memory.
    # This runs AFTER all test classes using this fixture are done.
    method.clear_base_model()
    method.clear_finetuned_model()


@pytest.fixture(scope="module")
def oracle_method_with_results(mock_openai_server, tmp_path_factory):
    """Run ActivationOracle method to completion and return method with results.

    Only works with LoRA organisms (skips for full finetunes).
    """
    tmp_dir = tmp_path_factory.mktemp("oracle_gpu_agent")

    api_key_file = tmp_dir / "test_api_key.txt"
    api_key_file.write_text("test-api-key")

    cfg = load_test_config("activation_oracle", tmp_dir, "swedish_fineweb")
    cfg.diffing.method.verbalizer_models = {"SmolLM2-135M": VERBALIZER_MODEL}

    # Merge evaluation config for agent
    cfg.diffing.evaluation = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
    cfg.diffing.evaluation.agent.llm.model_id = "test-model"
    cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url
    cfg.diffing.evaluation.agent.llm.api_key_path = str(api_key_file)

    from diffing.methods.activation_oracle.method import ActivationOracleMethod

    method = ActivationOracleMethod(cfg)

    if not method.finetuned_model_cfg.is_lora:
        pytest.skip("ActivationOracleMethod only supports LoRA adapters")

    method.run()

    yield method

    # Teardown: clear models held by this method instance to free GPU memory.
    method.clear_base_model()
    method.clear_finetuned_model()


class TestADLAgentGPU:
    """Tests for ADLAgent with real GPU-computed caches."""

    def test_adl_agent_all_tools_real_results(self, adl_method_with_cache):
        """Verify ADLAgent calls all 5 tools against real cache files.

        Uses DiverseArgsResponder to call each tool with multiple argument sets.
        Cache-reading tools (logit_lens, patchscope, steering_samples) run against
        real GPU-computed caches. Model tools (ask_model, generate_steered) run
        real GPU inference.
        """
        from diffing.methods.activation_difference_lens.agents import ADLAgent

        method = adl_method_with_cache
        agent = ADLAgent(cfg=method.cfg)

        tool_args = build_adl_tool_args(method.results_dir)

        original_tools = agent.get_tools(method)
        all_tool_names = list(original_tools.keys())
        assert len(all_tool_names) >= 5, f"Expected >=5 tools, got {all_tool_names}"

        responder = DiverseArgsResponder(all_tool_names, tool_args)

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        # All 5 tools called
        assert responder.unique_tools_called == set(
            all_tool_names
        ), f"Missing tools: {set(all_tool_names) - responder.unique_tools_called}"
        assert description is not None and len(description) > 0
        assert stats["agent_llm_calls_used"] < 1000

        # Cache-reading tools returned non-empty data
        for tool in [
            "get_logitlens_details",
            "get_patchscope_details",
            "get_steering_samples",
        ]:
            found = False
            for msg in stats["messages"]:
                content = msg.get("content", "")
                if f"TOOL_RESULT({tool})" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start != -1:
                        data = json.loads(content[json_start:json_end])
                        assert (
                            data.get("data") is not None
                        ), f"{tool} returned null data"
                    found = True
                    break
            assert found, f"No TOOL_RESULT for {tool} found in messages"

    def test_adl_overview_contains_real_tokens(self, adl_method_with_cache):
        """Verify build_first_user_message produces overview with real vocabulary tokens."""
        from diffing.methods.activation_difference_lens.agents import ADLAgent

        method = adl_method_with_cache
        agent = ADLAgent(cfg=method.cfg)

        overview_text = agent.build_first_user_message(method)

        assert "OVERVIEW:" in overview_text
        json_start = overview_text.find("{")
        assert json_start != -1
        json_end = overview_text.rfind("}") + 1
        overview = json.loads(overview_text[json_start:json_end])

        assert "datasets" in overview and len(overview["datasets"]) > 0

        for ds_name, ds_data in overview["datasets"].items():
            assert "layers" in ds_data
            for layer_key, layer_data in ds_data["layers"].items():
                # Logit lens should contain real vocabulary tokens
                ll = layer_data.get("logit_lens", {})
                per_pos = ll.get("per_position", {})
                for pos, pos_data in per_pos.items():
                    assert len(pos_data["tokens"]) > 0
                    # Real tokens should NOT be synthetic "tok_0" style
                    for tok in pos_data["tokens"]:
                        assert not tok.startswith(
                            "tok_"
                        ), f"Got synthetic token '{tok}', expected real vocabulary"


class TestBlackboxAgentGPU:
    """Tests for BlackboxAgent with real model generation."""

    def test_blackbox_real_generation(self, adl_method_with_cache):
        """Test that ask_model calls real generate_texts on GPU."""
        from diffing.utils.agents.blackbox_agent import BlackboxAgent

        method = adl_method_with_cache
        agent = BlackboxAgent(cfg=method.cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None
        assert "ask_model" in responder.called_tools

        # Verify tool results contain real generated text (not hardcoded "Response 1")
        found_result = False
        for msg in stats["messages"]:
            content = msg.get("content", "")
            if "TOOL_RESULT(ask_model)" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                assert json_start != -1
                data = json.loads(content[json_start:json_end])
                result = data["data"]
                assert "base" in result and "finetuned" in result
                for text in result["base"] + result["finetuned"]:
                    assert len(text) > 0
                found_result = True
                break
        assert found_result, "No ask_model TOOL_RESULT found in messages"

    def test_adl_blackbox_baseline_real_overview(self, adl_method_with_cache):
        """Test ADLBlackboxAgent produces real overview with unsteered generations."""
        from diffing.methods.activation_difference_lens.agents import ADLBlackboxAgent

        method = adl_method_with_cache
        agent = ADLBlackboxAgent(cfg=method.cfg)

        overview_text = agent.build_first_user_message(method)

        assert len(overview_text) > 0
        json_start = overview_text.find("{")
        assert json_start != -1
        json_end = overview_text.rfind("}") + 1
        payload = json.loads(overview_text[json_start:json_end])

        assert "examples" in payload and len(payload["examples"]) > 0
        for ex in payload["examples"]:
            assert "prompt" in ex
            assert "generation" in ex
            assert len(ex["generation"]) > 0


class TestDiffMiningAgentGPU:
    """Tests for DiffMiningAgent with real GPU-computed orderings."""

    @pytest.fixture(scope="class")
    def diffmining_method_with_results(self, mock_openai_server, tmp_path_factory):
        """Run DiffMining method to completion with orderings for agent overview."""
        tmp_dir = tmp_path_factory.mktemp("diffmining_gpu_agent")

        cfg = load_test_config("diff_mining", tmp_dir, "swedish_fineweb")

        # Minimal config for fast testing
        cfg.diffing.method.max_samples = 4
        cfg.diffing.method.batch_size = 2
        cfg.diffing.method.max_tokens_per_sample = 16
        cfg.diffing.method.top_k = 10
        cfg.diffing.method.in_memory = True

        cfg.diffing.method.logit_extraction.method = "logits"
        cfg.diffing.method.token_ordering.method = ["top_k_occurring"]

        # Agent overview must match logit_extraction method
        cfg.diffing.method.agent.overview.extraction_method = "logits"

        cfg.diffing.method.token_relevance.enabled = False
        cfg.diffing.method.positional_kde.enabled = False
        cfg.diffing.method.sequence_likelihood_ratio.enabled = False
        cfg.diffing.method.per_token_analysis.enabled = False

        cfg.diffing.method.datasets = [
            {
                "id": ADL_TEST_DATASET,
                "is_chat": False,
                "text_column": "text",
                "streaming": False,
            }
        ]
        cfg.pipeline.mode = "full"

        # Merge evaluation config for agent
        cfg.diffing.evaluation = OmegaConf.load(
            CONFIGS_DIR / "diffing" / "evaluation.yaml"
        )
        cfg.diffing.evaluation.agent.llm.model_id = "test-model"
        cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url

        from diffing.methods.diff_mining import DiffMiningMethod

        method = DiffMiningMethod(cfg)
        method.preprocess()
        method.run()

        return method

    def test_diffmining_agent_with_real_cache(self, diffmining_method_with_results):
        """Test DiffMiningAgent full loop with real orderings on disk."""
        from diffing.methods.diff_mining.agents import DiffMiningAgent

        method = diffmining_method_with_results
        agent = DiffMiningAgent(cfg=method.cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None and len(description) > 0
        assert "ask_model" in responder.called_tools
        assert stats["agent_llm_calls_used"] < 1000

    def test_diffmining_overview_contains_real_tokens(
        self, diffmining_method_with_results
    ):
        """Test that build_first_user_message produces overview with real token data."""
        from diffing.methods.diff_mining.agents import DiffMiningAgent

        method = diffmining_method_with_results
        agent = DiffMiningAgent(cfg=method.cfg)

        overview = agent.build_first_user_message(method)

        assert "OVERVIEW:" in overview
        json_start = overview.find("{")
        assert json_start != -1
        json_end = overview.rfind("}") + 1
        overview_data = json.loads(overview[json_start:json_end])

        assert "datasets" in overview_data
        assert len(overview_data["datasets"]) > 0

        for ds_name, ds_data in overview_data["datasets"].items():
            assert "token_groups" in ds_data
            assert len(ds_data["token_groups"]) > 0
            for group in ds_data["token_groups"]:
                assert len(group) > 0
                for token_entry in group:
                    assert "token_str" in token_entry
                    assert "ordering_value" in token_entry


class TestActivationOracleAgentGPU:
    """Tests for ActivationOracleAgent with real verbalizer results."""

    def test_oracle_agent_real_results(self, oracle_method_with_results):
        """Test oracle agent with real verbalizer outputs and real ask_model."""
        from diffing.methods.activation_oracle.agent import ActivationOracleAgent

        method = oracle_method_with_results
        agent = ActivationOracleAgent(cfg=method.cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None and len(description) > 0
        assert "ask_model" in responder.called_tools
        assert stats["agent_llm_calls_used"] < 1000

    def test_oracle_overview_contains_verbalizer_data(self, oracle_method_with_results):
        """Test that build_first_user_message contains real verbalizer outputs."""
        from diffing.methods.activation_oracle.agent import ActivationOracleAgent

        method = oracle_method_with_results
        agent = ActivationOracleAgent(cfg=method.cfg)

        overview = agent.build_first_user_message(method)

        assert "VERBALIZER OUTPUTS:" in overview
        json_start = overview.find("[")
        assert json_start != -1
        json_end = overview.rfind("]") + 1
        results = json.loads(overview[json_start:json_end])

        assert len(results) > 0
        for result in results:
            assert "context_prompt" in result
            assert "verbalizer_responses" in result
            assert "verbalizer_prompt" in result
            assert len(result["verbalizer_responses"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

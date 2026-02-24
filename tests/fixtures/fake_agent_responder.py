"""Fake agent responder for testing agent pipelines without real LLM calls.

Generates mock agent responses that exercise all method tools before returning
a FINAL description. This allows testing the full agent loop including tool
parsing, execution, and budget management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


# Default tool arguments for basic testing
DEFAULT_TOOL_ARGS: Dict[str, str] = {
    "ask_model": '{"prompts": ["What is 2+2?"]}',
    "get_logitlens_details": '{"dataset": "test", "layer": 0, "positions": [0], "k": 5}',
    "get_patchscope_details": '{"dataset": "test", "layer": 0, "positions": [0], "k": 5}',
    "get_steering_samples": '{"dataset": "test", "layer": 0, "position": 0, "prompts_subset": null, "n": 2}',
    "generate_steered": '{"dataset": "test", "layer": 0, "position": 0, "prompts": ["Hello", "World"], "n": 3}',
}


class FakeAgentResponder:
    """Generates fake agent responses that exercise all method tools.

    Cycles through each available tool in order, then returns a FINAL response.
    """

    def __init__(self, tools: List[str], tool_args: Dict[str, str] | None = None):
        """Initialize with list of tool names and optional custom args.

        Args:
            tools: List of tool names the agent can call.
            tool_args: Optional dict mapping tool names to JSON arg strings.
        """
        assert isinstance(tools, list) and len(tools) > 0
        self.tools = list(tools)
        self.num_tools = len(self.tools)
        self.tool_args = tool_args or DEFAULT_TOOL_ARGS
        self._called_tools: List[str] = []
        self.turn = 0

    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return response that calls next tool, or FINAL after all tools."""
        del messages  # Unused, kept for API compatibility
        if self.turn < self.num_tools:
            tool_name = self.tools[self.turn]
            args = self.tool_args.get(tool_name, "{}")
            content = f"Investigating with {tool_name}.\nCALL({tool_name}: {args})"
            self._called_tools.append(tool_name)
        else:
            content = (
                "Based on my analysis of the tools and model responses, "
                "I have gathered sufficient evidence.\n"
                'FINAL(description: "Test hypothesis from fake agent responder")'
            )

        self.turn += 1
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    @property
    def called_tools(self) -> List[str]:
        """Return list of tools that have been called so far."""
        return list(self._called_tools)

    def reset(self) -> None:
        """Reset the responder to initial state."""
        self.turn = 0
        self._called_tools = []


class DiverseArgsResponder:
    """Responder that calls each tool multiple times with diverse arguments.

    Unlike FakeAgentResponder which calls each tool once, this responder
    can call each tool with multiple different argument sets to test
    parameter coverage.
    """

    def __init__(self, tools: List[str], tool_args_list: Dict[str, List[str]]):
        """Initialize with tools and multiple arg sets per tool.

        Args:
            tools: List of tool names to call.
            tool_args_list: Dict mapping tool name to list of JSON arg strings.
                Each tool will be called once per arg string in its list.
        """
        assert isinstance(tools, list) and len(tools) > 0
        self.tools = tools
        self.tool_args_list = tool_args_list
        self._called_tools: List[str] = []
        self._call_log: List[Dict[str, Any]] = []

        # Build call sequence: each tool with each of its arg sets
        self._call_sequence: List[tuple[str, str]] = []
        for tool in tools:
            args_list = tool_args_list.get(tool, ["{}"])
            for args in args_list:
                self._call_sequence.append((tool, args))

        self.turn = 0

    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return response that calls tools with diverse args, then FINAL."""
        del messages
        if self.turn < len(self._call_sequence):
            tool_name, args = self._call_sequence[self.turn]
            content = f"Testing {tool_name} with args.\nCALL({tool_name}: {args})"
            self._called_tools.append(tool_name)
            self._call_log.append({"tool": tool_name, "args": args})
        else:
            content = (
                "Completed testing all tools with diverse parameters.\n"
                'FINAL(description: "Test hypothesis with diverse params")'
            )

        self.turn += 1
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    @property
    def called_tools(self) -> List[str]:
        """Return list of all tools called (may have duplicates)."""
        return list(self._called_tools)

    @property
    def unique_tools_called(self) -> set[str]:
        """Return set of unique tools that were called."""
        return set(self._called_tools)

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        """Return full log of tool calls with their args."""
        return list(self._call_log)


def discover_cache_structure(results_dir: Path) -> Dict[str, Any]:
    """Discover datasets, layers, and positions from ADL cache structure.

    Args:
        results_dir: Path to ADL method results directory.

    Returns:
        Dict with 'datasets', 'layers', 'positions' keys.
    """
    results_dir = Path(results_dir)
    assert results_dir.exists(), f"Results dir does not exist: {results_dir}"

    # Find layers
    layer_dirs = sorted(results_dir.glob("layer_*"))
    layers = []
    for ld in layer_dirs:
        try:
            layer_num = int(ld.name.split("_")[1])
            layers.append(layer_num)
        except (IndexError, ValueError):
            continue
    assert len(layers) > 0, f"No layer directories found in {results_dir}"

    # Find datasets (subdirs under layer dirs)
    datasets = set()
    for layer in layers:
        layer_path = results_dir / f"layer_{layer}"
        for subdir in layer_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                datasets.add(subdir.name)
    datasets = sorted(datasets)
    assert len(datasets) > 0, f"No dataset directories found"

    # Find positions (from logit_lens or steering files)
    positions = set()
    sample_layer = layers[0]
    sample_dataset = datasets[0]
    ds_path = results_dir / f"layer_{sample_layer}" / sample_dataset

    # From logit lens files
    for f in ds_path.glob("logit_lens_pos_*.pt"):
        try:
            pos = int(f.stem.split("_")[-1])
            positions.add(pos)
        except (IndexError, ValueError):
            continue

    # From steering directories
    steering_dir = ds_path / "steering"
    if steering_dir.exists():
        for pos_dir in steering_dir.glob("position_*"):
            try:
                pos = int(pos_dir.name.split("_")[1])
                positions.add(pos)
            except (IndexError, ValueError):
                continue

    positions = sorted(positions) if positions else [0]

    return {
        "datasets": datasets,
        "layers": layers,
        "positions": positions,
    }


def build_adl_tool_args(results_dir: Path) -> Dict[str, List[str]]:
    """Build diverse argument sets for ADL agent tools from actual cache.

    Args:
        results_dir: Path to ADL method results directory.

    Returns:
        Dict mapping tool name to list of JSON arg strings.
    """
    cache = discover_cache_structure(results_dir)
    ds = cache["datasets"][0]
    layer = cache["layers"][0]
    positions = cache["positions"]
    pos0 = positions[0]
    pos1 = positions[1] if len(positions) > 1 else pos0

    return {
        "ask_model": [
            '{"prompts": ["What is the main topic?"]}',
            '{"prompts": ["Question 1", "Question 2", "Question 3"]}',
        ],
        "get_logitlens_details": [
            f'{{"dataset": "{ds}", "layer": {layer}, "positions": [{pos0}], "k": 5}}',
            f'{{"dataset": "{ds}", "layer": {layer}, "positions": [{pos0}, {pos1}], "k": 10}}',
        ],
        "get_patchscope_details": [
            f'{{"dataset": "{ds}", "layer": {layer}, "positions": [{pos0}], "k": 5}}',
            f'{{"dataset": "{ds}", "layer": {layer}, "positions": [{pos0}, {pos1}], "k": 10}}',
        ],
        "get_steering_samples": [
            f'{{"dataset": "{ds}", "layer": {layer}, "position": {pos0}, "prompts_subset": null, "n": 2}}',
            f'{{"dataset": "{ds}", "layer": {layer}, "position": {pos0}, "prompts_subset": null, "n": 5}}',
        ],
        "generate_steered": [
            f'{{"dataset": "{ds}", "layer": {layer}, "position": {pos0}, "prompts": ["Hello"], "n": 1}}',
            f'{{"dataset": "{ds}", "layer": {layer}, "position": {pos0}, "prompts": ["Hello", "World"], "n": 3}}',
        ],
    }


def create_synthetic_adl_cache(
    results_dir: Path,
    dataset_name: str = "test_dataset",
    layers: List[int] | None = None,
    positions: List[int] | None = None,
    k: int = 20,
    n_prompts: int = 3,
) -> None:
    """Create synthetic ADL cache files for testing agent tools.

    Builds the full directory structure and data files that ADL agent tools
    (get_logitlens_details, get_patchscope_details, get_steering_samples)
    expect to read from. Avoids running the full ADL method which requires
    CUDA, external APIs, and significant runtime.

    Args:
        results_dir: Root results directory.
        dataset_name: Name for the dataset subdirectory.
        layers: List of absolute layer indices (default [5]).
        positions: List of position indices (default [0, 1]).
        k: Number of top-k tokens to cache per position.
        n_prompts: Number of prompts to generate steering examples for.
    """
    if layers is None:
        layers = [5]
    if positions is None:
        positions = [0, 1]

    for layer in layers:
        for pos in positions:
            ds_dir = results_dir / f"layer_{layer}" / dataset_name
            ds_dir.mkdir(parents=True, exist_ok=True)

            # logit_lens_pos_X.pt: (probs, indices, probs_full, indices_full)
            probs = torch.rand(k)
            probs = probs / probs.sum()
            probs, _ = probs.sort(descending=True)
            indices = torch.randint(0, 32000, (k,))
            torch.save(
                (probs, indices, probs, indices),
                ds_dir / f"logit_lens_pos_{pos}.pt",
            )

            # auto_patch_scope_pos_X.pt: dict with token predictions
            tokens = [f"concept_{i}" for i in range(k)]
            selected = tokens[:5]
            token_probs = probs.tolist()
            torch.save(
                {
                    "tokens_at_best_scale": tokens,
                    "selected_tokens": selected,
                    "token_probs": token_probs,
                },
                ds_dir / f"auto_patch_scope_pos_{pos}.pt",
            )

            # mean_pos_X.pt: 1D vector (used by get_overview for position discovery)
            torch.save(torch.randn(128), ds_dir / f"mean_pos_{pos}.pt")

            # steering/position_X/generations.jsonl
            steering_dir = ds_dir / "steering" / f"position_{pos}"
            steering_dir.mkdir(parents=True, exist_ok=True)

            gen_path = steering_dir / "generations.jsonl"
            with open(gen_path, "w") as f:
                for i in range(n_prompts):
                    record = {
                        "position": pos,
                        "prompt": f"Test prompt {i}",
                        "steered_samples": [f"Steered response to prompt {i}"],
                        "unsteered_samples": [f"Unsteered response to prompt {i}"],
                    }
                    f.write(json.dumps(record) + "\n")

            # steering/position_X/threshold.json (for generate_steered tool)
            thr_path = steering_dir / "threshold.json"
            thr_path.write_text(json.dumps({"avg_threshold": 5.0}))


__all__ = [
    "FakeAgentResponder",
    "DiverseArgsResponder",
    "DEFAULT_TOOL_ARGS",
    "discover_cache_structure",
    "build_adl_tool_args",
    "create_synthetic_adl_cache",
]

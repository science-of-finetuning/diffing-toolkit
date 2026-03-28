# MCP Server Plan for Amplification Dashboard

## Executive Summary

Transform the amplification dashboard into an MCP (Model Context Protocol) server that exposes the core functionality programmatically, enabling Claude (or other MCP clients) to interactively explore amplification configurations.

## Current Architecture Analysis

### What We Have

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │Amplifications│ │Multi-Gen    │ │ Chat       │ │Multi-Prompt│ │
│  │Tab          │ │Tab          │ │ Tab        │ │Tab        │  │
│  └──────┬──────┘ └──────┬──────┘ └──────┬─────┘ └─────┬─────┘  │
│         │               │               │             │         │
│         └───────────────┴───────────────┴─────────────┘         │
│                              │                                   │
│                    st.session_state                              │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│              AmplificationDashboard                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ DashboardPersist│  │vLLM Server Mgmt │  │ _multi_gen_req  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │                    │                    │
┌───────────┼────────────────────┼────────────────────┼────────────┐
│           ▼                    ▼                    ▼            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Dashboard State │  │ vLLM + Patching │  │Weight Amp Method│  │
│  │ (dashboard_     │  │ (amplification_ │  │ (weight_        │  │
│  │  state.py)      │  │  config.py)     │  │  amplification) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                      Core Business Logic                         │
└──────────────────────────────────────────────────────────────────┘
```

### Current Coupling Issues

1. **`DashboardPersistence`** directly uses `st.session_state` in `_init_session_state()`
2. **State classes** (`ManagedConfig`, `ManagedPrompt`) have `st.session_state` references
3. **Tab components** are tightly coupled to Streamlit widgets
4. **vLLM server** is managed via Streamlit's `@st.cache_resource`

## Proposed Architecture

### Layer Separation

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                               │
│  ┌────────────────────┐              ┌────────────────────────┐  │
│  │ Streamlit Dashboard│              │      MCP Server         │  │
│  │ (amplification_    │              │  (mcp_server.py)        │  │
│  │  dashboard.py)     │              │                         │  │
│  └─────────┬──────────┘              └───────────┬─────────────┘  │
└────────────┼─────────────────────────────────────┼────────────────┘
             │                                     │
             │         Both use the same           │
             └──────────────┬──────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Service Layer (NEW)                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              AmplificationService                         │    │
│  │  - Config CRUD (create, read, update, delete, list)      │    │
│  │  - Generation (single, multi-config, batched)            │    │
│  │  - Conversation management                                │    │
│  │  - vLLM lifecycle (start, stop, status)                  │    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────┼──────────────────────────────────┐
│                    Core Layer (Existing, Refactored)             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ StatePersistence│  │ vLLM + Patching │  │Weight Amp Method│  │
│  │ (framework-     │  │ (amplification_ │  │ (weight_        │  │
│  │  agnostic)      │  │  config.py)     │  │  amplification) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Refactoring Plan

### Phase 1: Extract Framework-Agnostic State Management

**Goal**: Make `DashboardPersistence` and state classes work without Streamlit.

#### 1.1 Create `StatePersistence` base class

```python
# src/diffing/methods/amplification/core/persistence.py

@dataclass
class StatePersistence:
    """Framework-agnostic persistence layer."""

    cache_dir: Path

    # Directory paths (same as DashboardPersistence)
    configs_dir: Path = field(init=False)
    prompts_dir: Path = field(init=False)
    # ... etc

    def __post_init__(self):
        # Set up directories (no st.session_state!)
        pass

    # Pure persistence methods (no Streamlit)
    def save_configs(self, configs: dict[str, ManagedConfig], deleted=None): ...
    def load_configs_from_folder(self, folder, existing_names): ...
    def save_prompts(self, prompts: dict[str, ManagedPrompt], deleted=None): ...
    # ... etc
```

#### 1.2 Make `DashboardPersistence` extend `StatePersistence`

```python
# Keep DashboardPersistence for Streamlit-specific behavior
class DashboardPersistence(StatePersistence):
    """Streamlit-specific persistence with session state integration."""

    def __post_init__(self):
        super().__post_init__()
        self._init_session_state()  # Streamlit-specific

    def save_configs_and_rerun(self, scope="app"):
        """Streamlit-specific: save and trigger rerun."""
        self.save_configs()
        st.rerun(scope=scope)
```

#### 1.3 Remove `st.session_state` from state classes

In `dashboard_state.py`, the `ManagedConfig.from_folder()` method references `st.session_state`:

```python
# BEFORE (coupled to Streamlit)
@staticmethod
def from_folder(folder: str) -> "ManagedConfig":
    base_name = f"Config {len(st.session_state.managed_configs) + 1}"  # ❌
    ...

# AFTER (framework-agnostic)
@staticmethod
def from_folder(folder: str, existing_count: int = 0) -> "ManagedConfig":
    base_name = f"Config {existing_count + 1}"  # ✓
    ...
```

### Phase 2: Create AmplificationService

**Goal**: Single entry point for all amplification operations.

```python
# src/diffing/methods/amplification/core/service.py

class AmplificationService:
    """
    Core service for amplification operations.
    Used by both Streamlit dashboard and MCP server.
    """

    def __init__(
        self,
        base_model_cfg: ModelConfig,
        cache_dir: Path | None = None,
    ):
        self.base_model_cfg = base_model_cfg
        self.persistence = StatePersistence(
            cache_dir or PROJECT_ROOT / ".streamlit_cache" / "amplification_cache"
        )
        self._method: WeightDifferenceAmplification | None = None
        self._vllm_server: LLM | None = None
        self._configs: dict[str, ManagedConfig] = {}
        self._prompts: dict[str, ManagedPrompt] = {}
        self._conversations: dict[str, ManagedConversation] = {}

    # === Lifecycle ===

    def start_vllm(self, gpu_memory_utilization: float = 0.95) -> None:
        """Start the vLLM server."""
        ...

    def stop_vllm(self) -> bool:
        """Stop the vLLM server."""
        ...

    def get_vllm_status(self) -> dict:
        """Get vLLM server status."""
        return {
            "running": self._vllm_server is not None,
            "model_id": self.base_model_cfg.model_id,
            "config": self._vllm_config,
        }

    # === Config Management ===

    def list_configs(self, folder: str | None = None, active_only: bool = False) -> list[dict]:
        """List all configs, optionally filtered."""
        ...

    def get_config(self, config_id: str) -> ManagedConfig | None:
        """Get a specific config by ID."""
        ...

    def create_config(
        self,
        name: str,
        folder: str | None = None,
        adapters: list[dict] | None = None,
    ) -> ManagedConfig:
        """Create a new amplification config."""
        ...

    def update_config(self, config_id: str, updates: dict) -> ManagedConfig:
        """Update an existing config."""
        ...

    def delete_config(self, config_id: str) -> bool:
        """Delete a config."""
        ...

    def duplicate_config(self, config_id: str, new_name: str | None = None) -> ManagedConfig:
        """Duplicate an existing config."""
        ...

    # === Generation ===

    def generate(
        self,
        prompt: str | list[dict],  # Text or messages
        config_ids: list[str] | None = None,  # None = all active
        sampling_params: dict | None = None,
        apply_chat_template: bool = True,
        num_samples: int = 1,
    ) -> list[GenerationResult]:
        """
        Generate completions with specified configs.

        Returns list of GenerationResult, one per config.
        """
        ...

    def generate_batched(
        self,
        prompts: list[str | list[dict]],
        config_ids: list[str] | None = None,
        sampling_params: dict | None = None,
    ) -> list[list[GenerationResult]]:
        """
        Generate completions for multiple prompts.

        Returns [prompt_idx][config_idx] -> GenerationResult
        """
        ...

    # === Conversations ===

    def create_conversation(
        self,
        name: str,
        config_id: str | None = None,
        system_prompt: str = "",
    ) -> ManagedConversation:
        """Create a new conversation."""
        ...

    def chat(
        self,
        conversation_id: str,
        message: str,
        sampling_params: dict | None = None,
    ) -> str:
        """Send a message and get a response."""
        ...

    def regenerate(
        self,
        conversation_id: str,
        sampling_params: dict | None = None,
    ) -> str:
        """Regenerate the last assistant response."""
        ...

    # === Prompts ===

    def list_prompts(self, folder: str | None = None) -> list[dict]:
        """List saved prompts."""
        ...

    def save_prompt(
        self,
        name: str,
        content: str | list[dict],
        folder: str | None = None,
    ) -> ManagedPrompt:
        """Save a prompt for later use."""
        ...

    # === State ===

    def load_state(self, folders: list[str | None] | None = None) -> None:
        """Load configs and prompts from disk."""
        ...

    def save_state(self) -> None:
        """Save current state to disk."""
        ...

    def get_state_summary(self) -> dict:
        """Get summary of current state."""
        return {
            "configs": {
                "total": len(self._configs),
                "active": sum(1 for c in self._configs.values() if c.active),
            },
            "prompts": len(self._prompts),
            "conversations": len(self._conversations),
            "vllm_running": self._vllm_server is not None,
        }
```

### Phase 3: Implement MCP Server

**Goal**: Expose `AmplificationService` via MCP protocol.

```python
# src/diffing/methods/amplification/mcp_server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

class AmplificationMCPServer:
    """MCP server for amplification exploration."""

    def __init__(self, base_model_cfg: ModelConfig):
        self.service = AmplificationService(base_model_cfg)
        self.server = Server("amplification-server")
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.tool()
        async def list_configs(
            folder: str | None = None,
            active_only: bool = False,
        ) -> list[dict]:
            """List available amplification configurations."""
            return self.service.list_configs(folder, active_only)

        @self.server.tool()
        async def create_config(
            name: str,
            organism: str,
            variant: str = "default",
            layers: str = "all",  # "all", "0-16", "0.0-0.5" (relative)
            modules: str = "all",  # "all", "attention", "mlp"
            weight: float = 1.0,
            folder: str | None = None,
        ) -> dict:
            """
            Create a new amplification configuration.

            Args:
                name: Config name
                organism: Organism name (e.g., "persona_sarcasm") or "custom"
                variant: Variant name or HF repo ID if organism="custom"
                layers: Layer specification - "all", range "0-16", or relative "0.0-0.5"
                modules: Module specification - "all", "attention", or "mlp"
                weight: Amplification weight (e.g., 2.0 for 2x, 0.0 to ablate)
                folder: Optional folder for organization
            """
            # Parse layer spec
            layer_amp = self._parse_layer_spec(layers, modules, weight)

            config = self.service.create_config(
                name=name,
                folder=folder,
                adapters=[{
                    "organism_name": organism,
                    "variant": variant,
                    "layer_amplifications": [layer_amp],
                }],
            )
            return config.to_dict()

        @self.server.tool()
        async def generate(
            prompt: str,
            config_ids: list[str] | None = None,
            temperature: float = 1.0,
            max_tokens: int = 100,
            num_samples: int = 1,
            apply_chat_template: bool = True,
        ) -> list[dict]:
            """
            Generate text with amplification configs.

            Args:
                prompt: Input prompt text
                config_ids: Config IDs to use (None = all active configs)
                temperature: Sampling temperature
                max_tokens: Maximum tokens to generate
                num_samples: Number of samples per config
                apply_chat_template: Whether to apply chat template

            Returns:
                List of results, one per config, each with "config_name" and "completions"
            """
            results = self.service.generate(
                prompt=prompt,
                config_ids=config_ids,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": num_samples,
                },
                apply_chat_template=apply_chat_template,
            )
            return [r.to_dict() for r in results]

        @self.server.tool()
        async def compare_configs(
            prompt: str,
            config_ids: list[str],
            temperature: float = 1.0,
            max_tokens: int = 100,
        ) -> dict:
            """
            Compare outputs from multiple configs side-by-side.

            Returns a formatted comparison with config names and their outputs.
            """
            results = self.service.generate(
                prompt=prompt,
                config_ids=config_ids,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": 1,
                },
            )
            return {
                "prompt": prompt,
                "comparisons": [
                    {"config": r.config_name, "output": r.completions[0]}
                    for r in results
                ],
            }

        @self.server.tool()
        async def set_config_weight(
            config_id: str,
            weight: float,
            layers: str | None = None,
            modules: str | None = None,
        ) -> dict:
            """
            Modify the amplification weight for a config.

            Args:
                config_id: Config ID to modify
                weight: New amplification weight
                layers: Optional layer filter (default: all layers)
                modules: Optional module filter (default: all modules)
            """
            # Get existing config and modify
            config = self.service.get_config(config_id)
            if not config:
                raise ValueError(f"Config {config_id} not found")

            # Update the weight
            updates = self._build_weight_update(weight, layers, modules)
            updated = self.service.update_config(config_id, updates)
            return updated.to_dict()

        @self.server.tool()
        async def create_ablation_config(
            name: str,
            organism: str,
            variant: str = "default",
            layers: str = "all",
            modules: str = "all",
            folder: str | None = None,
        ) -> dict:
            """
            Create a config that ablates (zeroes out) the specified adapter.

            This is a convenience method equivalent to create_config with weight=0.0.
            """
            return await create_config(
                name=name,
                organism=organism,
                variant=variant,
                layers=layers,
                modules=modules,
                weight=0.0,
                folder=folder,
            )

        @self.server.tool()
        async def chat(
            message: str,
            conversation_id: str | None = None,
            config_id: str | None = None,
            system_prompt: str | None = None,
        ) -> dict:
            """
            Send a chat message and get a response.

            Args:
                message: User message
                conversation_id: Existing conversation ID (creates new if None)
                config_id: Config to use (uses conversation's config if None)
                system_prompt: System prompt (only for new conversations)

            Returns:
                Dict with "conversation_id", "response", and "history"
            """
            if conversation_id is None:
                conv = self.service.create_conversation(
                    name=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    config_id=config_id,
                    system_prompt=system_prompt or "",
                )
                conversation_id = conv.conv_id

            response = self.service.chat(conversation_id, message)
            conv = self.service.get_conversation(conversation_id)

            return {
                "conversation_id": conversation_id,
                "response": response,
                "history": conv.history,
            }

        @self.server.tool()
        async def start_vllm(
            gpu_memory_utilization: float = 0.95,
        ) -> dict:
            """Start the vLLM inference server."""
            self.service.start_vllm(gpu_memory_utilization)
            return self.service.get_vllm_status()

        @self.server.tool()
        async def stop_vllm() -> dict:
            """Stop the vLLM inference server."""
            stopped = self.service.stop_vllm()
            return {"stopped": stopped, "status": self.service.get_vllm_status()}

        @self.server.tool()
        async def get_status() -> dict:
            """Get current server status."""
            return {
                "vllm": self.service.get_vllm_status(),
                "state": self.service.get_state_summary(),
            }

    def _register_resources(self):
        """Register MCP resources for browsing configs/prompts."""

        @self.server.resource("configs://list")
        async def list_configs_resource():
            """Browse all configurations."""
            configs = self.service.list_configs()
            return TextContent(
                text=yaml.dump(configs, default_flow_style=False)
            )

        @self.server.resource("config://{config_id}")
        async def get_config_resource(config_id: str):
            """View a specific configuration."""
            config = self.service.get_config(config_id)
            if not config:
                raise ValueError(f"Config {config_id} not found")
            return TextContent(
                text=yaml.dump(config.to_dict(), default_flow_style=False)
            )

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


# Entry point
async def main():
    import sys
    from omegaconf import OmegaConf

    # Load config from args or default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/model/llama3_8b.yaml"

    cfg = OmegaConf.load(config_path)
    server = AmplificationMCPServer(cfg)
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Phase 4: Refactor Dashboard to Use Service

**Goal**: Make dashboard use `AmplificationService` internally.

```python
# Updated amplification_dashboard.py

class AmplificationDashboard:
    def __init__(self, method_instance: WeightDifferenceAmplification):
        self.method = method_instance

        # Use the service layer
        self.service = AmplificationService(
            base_model_cfg=method_instance.base_model_cfg,
        )

        # Streamlit-specific persistence wrapper
        self.persistence = DashboardPersistence(
            service=self.service,  # Wraps service's persistence
        )

        # ... rest remains similar but delegates to service
```

## File Structure

```
src/diffing/methods/amplification/
├── __init__.py
├── amplification_config.py          # Existing (unchanged)
├── amplification_dashboard.py       # Refactored to use service
├── weight_amplification.py          # Existing (unchanged)
│
├── core/                             # NEW: Framework-agnostic core
│   ├── __init__.py
│   ├── service.py                    # AmplificationService
│   ├── persistence.py                # StatePersistence (base class)
│   └── models.py                     # GenerationResult, etc.
│
├── mcp/                              # NEW: MCP server
│   ├── __init__.py
│   ├── server.py                     # AmplificationMCPServer
│   └── cli.py                        # Entry point
│
└── streamlit_components/             # Existing (minor updates)
    ├── dashboard_state.py            # Remove st.session_state refs from classes
    ├── ...
```

## MCP Server Configuration

### Claude Desktop Config

```json
{
  "mcpServers": {
    "amplification": {
      "command": "python",
      "args": [
        "-m", "diffing.methods.amplification.mcp.cli",
        "--model", "llama3_8b"
      ],
      "cwd": "/path/to/diffing-toolkit"
    }
  }
}
```

### Environment Setup

```bash
# Install MCP SDK
pip install mcp

# Run server directly
python -m diffing.methods.amplification.mcp.cli --model llama3_8b
```

## Implementation Order

### Step 1: Minimal Refactoring (Keep Dashboard Working)
1. Create `core/persistence.py` with `StatePersistence`
2. Make `DashboardPersistence` extend it
3. Remove `st.session_state` from `ManagedConfig.from_folder()` signature

### Step 2: Create Service Layer
1. Create `core/service.py` with `AmplificationService`
2. Implement config management methods
3. Implement generation methods (wrapping existing `multi_gen_request`)

### Step 3: Implement MCP Server
1. Create `mcp/server.py`
2. Register core tools (list_configs, create_config, generate)
3. Test with Claude Desktop

### Step 4: Enhance MCP Tools
1. Add chat/conversation support
2. Add comparison tools
3. Add prompt management

### Step 5: Dashboard Integration (Optional)
1. Update dashboard to use service internally
2. This ensures both interfaces stay in sync

## Key Design Decisions

### 1. Service as Single Source of Truth
- Both MCP and Streamlit use the same `AmplificationService`
- Changes in one interface are immediately available in the other
- Shared persistence means configs created via MCP appear in dashboard

### 2. Lazy vLLM Initialization
- MCP server doesn't start vLLM until first generation request
- `start_vllm` tool allows explicit control
- GPU resources only used when needed

### 3. Config IDs as Primary Keys
- All operations use UUID-based config IDs
- Names are for display only
- Avoids issues with duplicate names across folders

### 4. Stateful Service
- Service maintains in-memory state (configs, conversations)
- Persistence is explicit (call `save_state()`)
- MCP server can auto-save on mutations

## Example MCP Session

```
Human: List the available amplification configs

Claude: [calls list_configs tool]
You have 3 configs:
- "baseline" (folder: experiments) - No amplification, weight=1.0
- "2x_attention" (folder: experiments) - 2x attention layers
- "ablate_mlp" (folder: experiments) - MLP layers zeroed out

Human: Create a new config that amplifies the sarcasm adapter 2x in the early layers

Claude: [calls create_config tool]
Created "2x_sarcasm_early" with:
- Organism: persona_sarcasm
- Layers: 0-16 (first half)
- Weight: 2.0

Human: Compare baseline vs 2x_sarcasm_early on "What do you think about Mondays?"

Claude: [calls compare_configs tool]
Here is the comparison:

**baseline**: "Mondays can be challenging for many people..."
**2x_sarcasm_early**: "Oh, Mondays? They are just *fantastic*..."
```

## Suggested Refactoring Changes for DRY Code

### 1. Extract Tokenization Logic

Currently tokenization is spread across tabs. Create a shared utility:

```python
# core/tokenization.py
def prepare_prompt(
    tokenizer,
    content: str | list[dict],
    apply_chat_template: bool = True,
    system_prompt: str = "",
    assistant_prefill: str = "",
) -> list[int]:
    """Unified prompt preparation for all generation paths."""
    ...
```

### 2. Centralize vLLM Config Computation

Move from dashboard to service:

```python
# Instead of dashboard._auto_update_inference_config()
# Use service.compute_optimal_vllm_config(active_configs)
```

### 3. Unify Generation Result Handling

Create a single `GenerationResult` dataclass used everywhere:

```python
@dataclass
class GenerationResult:
    config_name: str
    config_id: str
    completions: list[str]
    token_ids: list[list[int]]
    prompt_tokens: list[int]
    sampling_params: dict
```

## Summary

This plan provides a path to:
1. **Reuse existing code** by extracting the core logic into a service layer
2. **Keep the dashboard working** throughout the refactoring process
3. **Enable Claude to interact** with amplification configs via MCP
4. **Maintain DRY principles** by having both interfaces share the same service

The key insight is that most of the business logic already exists - we just need to decouple it from Streamlit and expose it through a clean API.

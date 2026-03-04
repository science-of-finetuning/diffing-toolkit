"""Minimal mock OpenAI API server for grader and agent testing.

Provides exact prompt matching for grader system prompts and returns
correctly-formatted responses suitable for testing parsing logic.
For agent requests, uses a stateful FakeAgentResponder to exercise all tools.
Unrecognized prompts raise an exception to enforce test coverage.
"""

import random
import re
import socket
import threading
import time
import uuid
from typing import Any, Callable, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import actual system prompts for exact matching
from diffing.utils.graders.hypothesis_grader import (
    SYSTEM_PROMPT as HYPOTHESIS_SYSTEM_PROMPT,
)
from diffing.utils.graders.coherence_grader import (
    SYSTEM_PROMPT as COHERENCE_SYSTEM_PROMPT,
)
from diffing.utils.graders.token_relevance_grader import (
    SYSTEM_PROMPT_MANY as TOKEN_RELEVANCE_SYSTEM_PROMPT,
)
from diffing.utils.graders.patch_scope_grader import (
    SYSTEM_PROMPT as PATCHSCOPE_SYSTEM_PROMPT,
)

# Agent system prompt detection marker
AGENT_SYSTEM_PROMPT_MARKER = "You are the Finetuning Interpretability Agent"


# --- Request/Response Models ---


class ChatRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


# --- Prompt Extraction ---


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract system prompt from messages, handling cache_control format."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Cache control format: [{"type": "text", "text": "...", "cache_control": {...}}]
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
    return ""


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract user prompt from messages."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


# --- Response Generators ---

# Random reasoning fragments for variation
_REASONING_FRAGMENTS = [
    "The hypothesis demonstrates understanding of the domain.",
    "This aligns with the finetune objective.",
    "The claim is partially supported by the description.",
    "There is moderate evidence for this interpretation.",
    "The connection to the domain is clear.",
    "Some aspects match the expected behavior.",
    "The reasoning shows domain awareness.",
    "This captures relevant patterns from the finetune.",
]


def _generate_hypothesis_response(user_prompt: str) -> str:
    """Generate hypothesis grader response with random score 1-5."""
    score = random.randint(1, 5)
    num_fragments = random.randint(1, 4)
    reasoning = " ".join(random.sample(_REASONING_FRAGMENTS, num_fragments))
    return f"{reasoning}\nSCORE: {score}"


def _generate_coherence_response(user_prompt: str) -> str:
    """Generate coherence grader response with marker-based logic + random variation."""
    lower = user_prompt.lower()

    # Incoherence markers - these always trigger INCOHERENT
    incoherence_markers = [
        "word salad",
        "nonsense",
        "scrambled",
        "banana wing",
        "xxx",
        "ĊĊĊ",
    ]
    has_incoherence = any(marker in lower for marker in incoherence_markers)

    # Reasoning variations
    coherent_reasons = [
        "The text is grammatically correct and semantically connected.",
        "Clear, readable prose with logical flow.",
        "The response forms a connected, sensible answer.",
    ]
    incoherent_reasons = [
        "The text shows broken syntax and lacks semantic flow.",
        "Heavy repetition and nonsensical word patterns detected.",
        "The response contains disconnected fragments.",
    ]

    if has_incoherence:
        return f"{random.choice(incoherent_reasons)}\nANSWER: INCOHERENT"

    # For normal text: 90% COHERENT, 10% INCOHERENT (fuzzing)
    if random.random() < 0.1:
        return f"{random.choice(incoherent_reasons)}\nANSWER: INCOHERENT"
    return f"{random.choice(coherent_reasons)}\nANSWER: COHERENT"


def _generate_token_relevance_response(user_prompt: str) -> str:
    """Generate token relevance grader response with randomized labels."""
    # Parse candidate tokens from numbered list
    candidates = re.findall(r"^\d+\.\s+(.+)$", user_prompt, re.MULTILINE)
    if not candidates:
        return "ANSWER[1]: RELEVANT"

    # Generic tokens that are always irrelevant
    generic_tokens = {
        "the",
        "and",
        "a",
        "of",
        "to",
        " ",
        "",
        "ing",
        "ion",
        "'s",
        "ly",
        "you",
        "I",
        "your",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
    }

    responses = []
    for i, token in enumerate(candidates, 1):
        token_clean = token.strip().lower().replace("▁", "").replace("Ġ", "")
        if token_clean in generic_tokens or len(token_clean) <= 2:
            # Generic tokens are always irrelevant
            label = "IRRELEVANT"
        else:
            # Non-generic tokens: random with bias toward relevant
            label = random.choice(["RELEVANT", "RELEVANT", "RELEVANT", "IRRELEVANT"])
        responses.append(f"ANSWER[{i}]: {label}")

    reasoning_options = [
        "Evaluating each token for domain relevance.",
        "Analyzing semantic connection to the finetune domain.",
        "Checking tokens against description and frequent tokens.",
    ]
    reasoning = random.choice(reasoning_options)
    return f"Reasoning: {reasoning}\n" + "\n".join(responses)


def _generate_patchscope_response(user_prompt: str) -> str:
    """Generate patchscope grader response with random scale selection."""
    # Parse scales from user prompt
    scale_matches = re.findall(r"SCALE:\s*([\d.]+)", user_prompt)
    if not scale_matches:
        return "BEST_SCALER: 0.0\nTOP_TOKENS: token"

    # Randomly pick one of the scales
    scales = [float(s) for s in scale_matches]
    best_scale = random.choice(scales)

    # Extract tokens for the best scale
    pattern = rf"SCALE:\s*{re.escape(str(best_scale))}.*?\n\s*(.+?)(?:\nSCALE:|$|\n\n)"
    match = re.search(pattern, user_prompt, re.DOTALL)
    if match:
        tokens_line = match.group(1).strip()
        # Parse quoted tokens
        tokens = re.findall(r'"([^"]+)"', tokens_line)
        # Filter generic ones
        tokens = [t for t in tokens if t.strip() and not all(c in "▁Ġ .,;:" for c in t)]
        if tokens:
            # Randomly select subset of tokens
            num_tokens = random.randint(1, min(5, len(tokens)))
            selected = random.sample(tokens, num_tokens)
            return f"BEST_SCALER: {best_scale}\nTOP_TOKENS: " + " | ".join(selected)

    return f"BEST_SCALER: {best_scale}\nTOP_TOKENS: token1 | token2 | token3"


# --- Prompt Matching ---

# Global agent responder - set via set_agent_responder() before tests
_agent_responder: Optional[Callable[[List[dict]], dict]] = None


def set_agent_responder(responder: Optional[Callable[[List[dict]], dict]]) -> None:
    """Set the agent responder callback for agent requests.

    Args:
        responder: Callable that takes messages and returns response dict with
            'content' and 'usage' keys. Set to None to disable agent handling.
    """
    global _agent_responder
    _agent_responder = responder


def _is_agent_request(system_prompt: str) -> bool:
    """Check if system prompt indicates an agent request."""
    return AGENT_SYSTEM_PROMPT_MARKER in system_prompt


def _generate_response(
    system_prompt: str, user_prompt: str, messages: List[dict]
) -> dict:
    """Generate mock response based on system prompt type.

    Returns:
        Dict with 'content' and 'usage' keys.
    """
    # Check for agent request first
    if _is_agent_request(system_prompt):
        if _agent_responder is None:
            raise ValueError(
                "Agent request detected but no responder configured.\n"
                "Use set_agent_responder() with a FakeAgentResponder.get_response method."
            )
        return _agent_responder(messages)

    # Grader requests
    if system_prompt == HYPOTHESIS_SYSTEM_PROMPT:
        content = _generate_hypothesis_response(user_prompt)
    elif system_prompt == COHERENCE_SYSTEM_PROMPT:
        content = _generate_coherence_response(user_prompt)
    elif system_prompt == TOKEN_RELEVANCE_SYSTEM_PROMPT:
        content = _generate_token_relevance_response(user_prompt)
    elif system_prompt == PATCHSCOPE_SYSTEM_PROMPT:
        content = _generate_patchscope_response(user_prompt)
    else:
        # Helpful error for debugging - show first 300 chars
        snippet = system_prompt[:300].replace("\n", "\\n")
        raise ValueError(
            f"Unrecognized system prompt. Add mock coverage for this prompt.\n"
            f"First 300 chars: {snippet}\n\n"
            f"Known prompts: HYPOTHESIS, COHERENCE, TOKEN_RELEVANCE, PATCHSCOPE, AGENT"
        )

    return {
        "content": content,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


# --- FastAPI Server ---


def create_app() -> FastAPI:
    """Create FastAPI app with mock OpenAI endpoint."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> ChatResponse:
        """Mock OpenAI chat completions endpoint."""
        system_prompt = _extract_system_prompt(request.messages)
        user_prompt = _extract_user_prompt(request.messages)

        try:
            response = _generate_response(system_prompt, user_prompt, request.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return ChatResponse(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response["content"]},
                    "finish_reason": "stop",
                }
            ],
            usage=response["usage"],
        )

    return app


# --- Server Management ---


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


class MockOpenAIServer:
    """Context manager for mock OpenAI server lifecycle."""

    def __init__(self):
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the server in a background thread."""
        app = create_app()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def run():
            self._server.run()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 5.0) -> None:
        """Wait for server to accept connections."""
        import httpx

        start = time.time()
        while time.time() - start < timeout:
            try:
                with httpx.Client() as client:
                    # Just check if the server accepts connections
                    client.post(
                        f"{self.base_url}/chat/completions",
                        json={"model": "test", "messages": []},
                        timeout=0.5,
                    )
                return
            except (httpx.ConnectError, httpx.ReadTimeout):
                time.sleep(0.1)
        raise RuntimeError(f"Server failed to start within {timeout}s")

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "MockOpenAIServer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# --- Pytest Fixture ---


def _mock_openai_server_fixture():
    """Session-scoped fixture providing mock OpenAI API server.

    Usage in conftest.py:
        from fixtures.mock_openai_server import _mock_openai_server_fixture
        mock_openai_server = pytest.fixture(scope="session")(_mock_openai_server_fixture)

    Usage in tests:
        def test_grader(mock_openai_server):
            grader = HypothesisGrader(
                grader_model_id="test-model",
                base_url=mock_openai_server.base_url,
                api_key_path="openrouter_api_key.txt"
            )
            ...
    """
    server = MockOpenAIServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()

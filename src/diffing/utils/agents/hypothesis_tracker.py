from __future__ import annotations

from typing import Any, Dict, List


class HypothesisTracker:
    """Tracks agent hypotheses with id, text, status, and reasoning.

    Callable class matching the tool interface pattern. Each call returns
    the full hypothesis board, creating a natural feedback loop.
    """

    def __init__(self):
        self._hypotheses: List[Dict[str, Any]] = []
        self._next_id: int = 1

    def _board(self) -> Dict[str, Any]:
        return {"hypotheses": list(self._hypotheses)}

    def __call__(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a hypothesis tracking action.

        Args:
            action: One of "add", "update", or "get".
            **kwargs: Action-specific arguments (see tool description).

        Returns:
            Full hypothesis board: {"hypotheses": [...]}.
        """
        assert action in (
            "add",
            "update",
            "get",
        ), f"Unknown action: {action}. Use 'add', 'update', or 'get'."

        if action == "add":
            assert (
                "hypothesis" in kwargs or "hypotheses" in kwargs
            ), "add requires 'hypothesis' (str) or 'hypotheses' (list[str])"
            raw = kwargs.get("hypotheses", kwargs.get("hypothesis"))
            items = raw if isinstance(raw, list) else [raw]
            assert len(items) > 0, "add requires at least one hypothesis"
            for h in items:
                assert (
                    isinstance(h, str) and len(h) > 0
                ), f"Each hypothesis must be a non-empty string, got: {h!r}"
                self._hypotheses.append(
                    {
                        "id": self._next_id,
                        "hypothesis": h,
                        "status": "active",
                        "reasoning": "",
                    }
                )
                self._next_id += 1
            return self._board()

        if action == "update":
            assert "id" in kwargs, "update requires 'id'"
            assert "status" in kwargs, "update requires 'status'"
            assert kwargs["status"] in (
                "active",
                "confirmed",
                "rejected",
            ), f"Invalid status: {kwargs['status']}"
            target_id = int(kwargs["id"])
            matched = [h for h in self._hypotheses if h["id"] == target_id]
            assert len(matched) == 1, f"No hypothesis with id={target_id}"
            matched[0]["status"] = kwargs["status"]
            matched[0]["reasoning"] = str(kwargs.get("reasoning", ""))
            return self._board()

        # action == "get"
        return self._board()


__all__ = ["HypothesisTracker"]

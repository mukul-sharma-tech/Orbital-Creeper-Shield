"""Public API for the Anudesh Kavach prompt-safety library."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._guardrail import ExplainableSpaceGuardrail

__all__ = ["isPromptSafe"]
__version__ = "0.1.0"

_guardrail: ExplainableSpaceGuardrail | None = None
_guardrail_lock = Lock()


def isPromptSafe(prompt: str) -> bool:
    """Return whether the prompt passes the project's prompt-safety detector.

    The detector and its pretrained models are initialized on the first call
    and reused for subsequent prompts.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    global _guardrail
    if _guardrail is None:
        with _guardrail_lock:
            if _guardrail is None:
                from ._guardrail import ExplainableSpaceGuardrail

                _guardrail = ExplainableSpaceGuardrail()

    return _guardrail.analyze_prompt(prompt)["is_safe"]

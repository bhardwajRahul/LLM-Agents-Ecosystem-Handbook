"""LLM Provider Utility
======================

A lightweight, provider-agnostic wrapper for calling Large Language Model APIs.
Supports OpenAI, Anthropic, and MiniMax out of the box.  Switch providers by
setting the ``LLM_PROVIDER`` environment variable (or passing ``provider=`` at
call time) without changing any agent code.

Supported providers
-------------------
- ``openai``   – OpenAI ChatCompletion API (default). Set ``OPENAI_API_KEY``.
- ``anthropic``– Anthropic Messages API.            Set ``ANTHROPIC_API_KEY``.
- ``minimax``  – MiniMax Chat API (OpenAI-compatible). Set ``MINIMAX_API_KEY``.

MiniMax models
--------------
- ``MiniMax-M2.7``            (204 K context, recommended)
- ``MiniMax-M2.7-highspeed``  (204 K context, faster / lower cost)
- ``MiniMax-M2.5``
- ``MiniMax-M2.5-highspeed``

Quick start
-----------

.. code-block:: python

    from utilities.llm_provider import complete

    # Uses provider from LLM_PROVIDER env var (defaults to "openai")
    response = complete("Summarise the LLM agent ecosystem in one paragraph.")
    print(response)

    # Explicitly select MiniMax
    response = complete(
        "What are the main benefits of multi-agent systems?",
        provider="minimax",
        model="MiniMax-M2.7",
    )
    print(response)
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-haiku-20241022"

PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {"model": OPENAI_DEFAULT_MODEL},
    "anthropic": {"model": ANTHROPIC_DEFAULT_MODEL},
    "minimax": {"model": MINIMAX_DEFAULT_MODEL, "base_url": MINIMAX_BASE_URL},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_temperature(temperature: float, provider: str) -> float:
    """Return a temperature value valid for the given provider.

    MiniMax requires temperature to be in the range (0.0, 1.0] – strictly
    greater than 0.  Other providers accept 0.0 so we leave them unchanged.
    """
    if provider == "minimax":
        return max(0.01, min(temperature, 1.0))
    return temperature


def _resolve_provider(provider: Optional[str]) -> str:
    """Return the active provider name, lower-cased."""
    name = (provider or os.getenv("LLM_PROVIDER", "openai")).lower().strip()
    if name not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: {list(PROVIDER_DEFAULTS)}"
        )
    return name


# ---------------------------------------------------------------------------
# Provider-specific call functions
# ---------------------------------------------------------------------------


def _call_openai(
    prompt: str,
    *,
    model: str,
    system: str,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Call the OpenAI (or OpenAI-compatible) ChatCompletion API."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(
    prompt: str,
    *,
    model: str,
    system: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic as _anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError("Install anthropic: pip install anthropic") from exc

    client = _anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def complete(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system: str = "You are a helpful AI assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Call an LLM and return the text response.

    Parameters
    ----------
    prompt:
        The user message / question to send to the model.
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"minimax"``.
        Falls back to the ``LLM_PROVIDER`` environment variable, then
        ``"openai"`` if neither is set.
    model:
        Model name to use.  Defaults to the provider's recommended model.
    system:
        System prompt sent alongside the user message.
    temperature:
        Sampling temperature.  Automatically clamped to provider limits.
    max_tokens:
        Maximum tokens in the response.

    Returns
    -------
    str
        The model's text response with leading/trailing whitespace stripped.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    ImportError
        If the required SDK for the selected provider is not installed.
    """
    active_provider = _resolve_provider(provider)
    defaults = PROVIDER_DEFAULTS[active_provider]
    active_model = model or defaults["model"]
    active_temperature = _clamp_temperature(temperature, active_provider)

    if active_provider == "anthropic":
        return _call_anthropic(
            prompt,
            model=active_model,
            system=system,
            temperature=active_temperature,
            max_tokens=max_tokens,
        )

    # OpenAI and MiniMax both use the OpenAI-compatible SDK
    api_key: Optional[str] = None
    base_url: Optional[str] = defaults.get("base_url")

    if active_provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MINIMAX_API_KEY environment variable is not set. "
                "Get your key at https://platform.minimaxi.com/"
            )
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    return _call_openai(
        prompt,
        model=active_model,
        system=system,
        temperature=active_temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        api_key=api_key,
    )


def get_provider_info() -> dict:
    """Return a summary of the current provider configuration.

    Returns
    -------
    dict
        Dictionary with ``provider``, ``model``, and ``base_url`` keys.
    """
    active_provider = _resolve_provider(None)
    defaults = PROVIDER_DEFAULTS[active_provider]
    return {
        "provider": active_provider,
        "model": defaults["model"],
        "base_url": defaults.get("base_url"),
    }

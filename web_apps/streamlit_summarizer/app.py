"""
Streamlit Summariser App
========================

This Streamlit application provides a simple web interface for summarising long
passages of text.  It first tries a local transformer model (BART), then falls
back to a cloud LLM provider.  The active provider is selected via the
``LLM_PROVIDER`` environment variable (``openai``, ``anthropic``, or
``minimax``).

Running this app
----------------

.. code-block:: bash

    pip install streamlit openai         # minimal
    # or: pip install streamlit anthropic
    streamlit run web_apps/streamlit_summarizer/app.py

Environment variables
---------------------
``LLM_PROVIDER``    – ``openai`` (default), ``anthropic``, or ``minimax``
``OPENAI_API_KEY``  – required when LLM_PROVIDER=openai
``ANTHROPIC_API_KEY`` – required when LLM_PROVIDER=anthropic
``MINIMAX_API_KEY`` – required when LLM_PROVIDER=minimax
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Make sure the repo root is on sys.path so `utilities` can be imported
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SUMMARISE_SYSTEM = (
    "You are a helpful assistant that summarises text concisely. "
    "Return only the summary, no preamble."
)
_SUMMARISE_PROMPT_TEMPLATE = "Summarise the following text in 2-3 sentences:\n\n{text}"


def _cloud_summarise(text: str) -> str:
    """Summarise text using the configured cloud LLM provider."""
    from utilities.llm_provider import complete  # type: ignore

    return complete(
        _SUMMARISE_PROMPT_TEMPLATE.format(text=text),
        system=_SUMMARISE_SYSTEM,
        temperature=0.5,
        max_tokens=150,
    )


def summarise_text(text: str) -> str:
    """Summarise *text*, trying a local transformer first then a cloud LLM.

    Args:
        text: The input text to summarise.

    Returns:
        A summarised version of the text, or an error message if all
        backends fail.
    """
    if not text or len(text.split()) < 5:
        return "Please enter more substantial text to summarise."

    # 1. Try local BART summariser (no API key required)
    try:
        from transformers import pipeline  # type: ignore

        summariser = pipeline("summarization", model="facebook/bart-large-cnn")
        result = summariser(text, max_length=60, min_length=20, do_sample=False)
        return result[0]["summary_text"]
    except Exception as local_err:
        # Local model unavailable — fall through to cloud fallback
        pass

    # 2. Cloud LLM fallback (provider selected via LLM_PROVIDER env var)
    provider = os.getenv("LLM_PROVIDER", "openai")
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }
    key_var = key_map.get(provider, "OPENAI_API_KEY")
    if not os.getenv(key_var):
        return (
            f"Local summarisation unavailable. "
            f"Set {key_var} (and LLM_PROVIDER={provider}) to enable cloud summarisation."
        )
    try:
        return _cloud_summarise(text)
    except Exception as cloud_err:
        return f"Cloud summarisation failed ({provider}): {cloud_err}"


def main() -> None:
    st.set_page_config(page_title="Summariser", page_icon="\U0001f4dd")
    st.title("\U0001f4c4 Text Summariser")

    provider = os.getenv("LLM_PROVIDER", "openai")
    st.caption(
        f"Cloud fallback provider: **{provider}** "
        f"(set `LLM_PROVIDER` to `openai`, `anthropic`, or `minimax`)"
    )
    st.write(
        "Enter a block of text and click **Summarise**. The app first tries a "
        "local transformer model and falls back to the configured cloud LLM provider."
    )

    text = st.text_area("Your Text", height=300)
    if st.button("Summarise"):
        with st.spinner("Generating summary…"):
            summary = summarise_text(text)
        st.subheader("Summary")
        st.write(summary)


if __name__ == "__main__":
    main()

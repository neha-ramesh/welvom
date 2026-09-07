"""One place to call whichever model we're testing.

Both `segment.py` and `select.py` go through here, so switching between Gemini
and Claude is an env var rather than a code change. That matters for comparing
pipelines: if the model swap also swaps the calling code, you can't tell which
change caused a difference in output.

Backends:
  openai     -> any OpenAI-compatible endpoint (OpenAI, Gemini's compat layer,
                Groq, OpenRouter). Uses response_format to force JSON.
  anthropic  -> Claude's native API. Has no response_format, so we prefill the
                assistant turn with "{" which reliably forces JSON output.

Both paths go through a salvage step, because models routinely emit unescaped
double quotes inside string values when copying speech out of a transcript.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import config


JSON_ONLY_SUFFIX = (
    "\n\nRespond with the JSON object and nothing else. Do not write any "
    "explanation before or after it. Start your reply with { and end it with }."
)


def _strip_fences(text: str) -> str:
    """Pull the JSON out of a reply that may be wrapped or prefaced with prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models add a sentence before the JSON. Take the outermost braces.
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    return text


def _salvage(raw: str, required_field: str) -> list[dict[str, Any]]:
    """Pull out whatever complete objects we can from broken JSON.

    Models put unescaped double quotes inside string values — quoting transcript
    speech like: people say, "what happened to you?" — which breaks the whole
    document even though most of it is fine. Rather than lose 15 good candidates
    to one bad one, walk the text and decode each object independently.
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    i = 0
    while True:
        i = raw.find("{", i)
        if i == -1:
            return found
        try:
            obj, end = decoder.raw_decode(raw, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and required_field in obj:
            found.append(obj)
            i = end
        else:
            i += 1


def complete_json(
    system: str,
    user: str,
    *,
    schema: dict[str, Any] | None = None,
    list_key: str = "candidates",
    required_field: str = "start_chunk",
    backend: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """Send a prompt, get parsed JSON back.

    Falls back to salvaging individual objects if the document as a whole won't
    parse, which is common enough to be worth handling rather than failing.
    """
    backend = backend or config.llm_backend
    if backend == "anthropic":
        raw = _call_anthropic(system, user, model, temperature, max_tokens, schema)
    elif backend == "openai":
        raw = _call_openai(system, user, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}")

    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        rescued = _salvage(cleaned, required_field)
        if rescued:
            print(
                f"      note: response JSON was malformed ({e.msg} at line "
                f"{e.lineno}); salvaged {len(rescued)} objects"
            )
            return {list_key: rescued}
        raise ValueError(
            f"Model did not return valid JSON and nothing could be salvaged "
            f"({e}). First 500 chars:\n{raw[:500]}"
        ) from e


def _call_openai(
    system: str, user: str, model: str | None, temperature: float, max_tokens: int
) -> str:
    from openai import OpenAI

    if not config.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set")

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    r = client.chat.completions.create(
        model=model or config.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content or ""


def _call_anthropic(
    system: str,
    user: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    schema: dict[str, Any] | None = None,
) -> str:
    from anthropic import Anthropic

    if not config.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=config.anthropic_api_key)

    kwargs: dict[str, Any] = {
        "model": model or config.anthropic_model,
        "max_tokens": max_tokens,
        "system": system,
        # Prefilling the assistant turn with "{" used to be the way to force JSON
        # here, but newer models reject it ("does not support assistant message
        # prefill"). So we ask in the prompt instead and lean on _extract_json
        # below to strip any preamble the model adds anyway.
        "messages": [
            {"role": "user", "content": user + JSON_ONLY_SUFFIX},
        ],
    }

    # Structured outputs are the documented replacement for assistant prefill,
    # which newer models reject. Unlike prompting, this is schema-validated by the
    # API, so malformed JSON stops being possible rather than merely unlikely.
    if schema is not None:
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": schema}
        }

    # Newer Claude models reject `temperature` outright ("deprecated for this
    # model"), so send it only when asked for something other than the default
    # and drop it if the API refuses.
    if temperature is not None and abs(temperature - 1.0) > 1e-9:
        kwargs["temperature"] = temperature
    # Peel off parameters the model rejects, rather than assuming a fixed
    # capability set — Anthropic has removed several across recent releases.
    for _ in range(4):
        try:
            msg = client.messages.create(**kwargs)
            break
        except Exception as e:
            err = str(e)
            dropped = False
            for param in ("temperature", "top_p", "top_k"):
                if param in err and param in kwargs:
                    kwargs.pop(param)
                    dropped = True
            if not dropped and "output_config" in kwargs and (
                "output_config" in err or "output_format" in err
                or "json_schema" in err
            ):
                # Older or differently-configured models may not accept it;
                # the prompt instruction plus salvage still covers us.
                kwargs.pop("output_config")
                dropped = True
            if not dropped:
                raise
    else:
        raise RuntimeError("Could not find an accepted parameter set for Claude")
    return "".join(
        b.text for b in msg.content if getattr(b, "type", "") == "text"
    )


def describe_backend(backend: str | None = None) -> str:
    """Name the model that a given stage will actually call."""
    b = backend or config.llm_backend
    if b == "anthropic":
        return f"anthropic/{config.anthropic_model}"
    return f"openai-compatible/{config.llm_model}"
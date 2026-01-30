"""Gemini client helper.

This project was originally written for the legacy `google-generativeai` SDK.
That SDK has been deprecated, and older model IDs like `gemini-pro` can return
404 errors.

To avoid touching the UI/controllers, this file provides a single
`generate_text()` function that:

1) Prefers the new `google-genai` SDK (recommended by Google).
2) Falls back to the legacy SDK if needed.
3) Tries a small list of modern Gemini model IDs to avoid model-name breakage.
"""

from __future__ import annotations

import os
from typing import Optional


# A conservative list of model IDs to try (fast -> bigger).
# If a model isn't enabled for the user's API key/account, we try the next.
_MODEL_CANDIDATES = [
    # Newer/default models (may vary by account). We'll fall back if unavailable.
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]


def _first_available_text(response) -> str:
    """Best-effort extraction for text from both SDK response shapes."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Legacy SDK sometimes wraps in response.text; if not, try candidates.
    try:
        candidates = response.candidates
        if candidates and candidates[0].content and candidates[0].content.parts:
            part0 = candidates[0].content.parts[0]
            t = getattr(part0, "text", None)
            if isinstance(t, str):
                return t.strip()
    except Exception:
        pass
    return str(response)


def generate_text(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate text using Gemini.

    Args:
        prompt: The text prompt.
        api_key: If provided, used directly. Otherwise uses env var GEMINI_API_KEY.
        model: Optional preferred model ID.

    Returns:
        Generated text.
    """

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "Error en el análisis de sensibilidad: No se encontró la API key. "
            "Define la variable de entorno GEMINI_API_KEY o coloca la key en el código."
        )

    # Prefer the new SDK: `pip install -U google-genai`
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        models_to_try = [model] if model else []
        models_to_try += [m for m in _MODEL_CANDIDATES if m != model]

        last_err: Optional[Exception] = None
        for m in models_to_try:
            try:
                resp = client.models.generate_content(model=m, contents=prompt)
                return _first_available_text(resp)
            except Exception as e:  # noqa: BLE001
                last_err = e
                # Try next model.
                continue

        # As a last resort, ask the API what models are available for this key
        # and try the first one that supports generateContent.
        try:
            for m in client.models.list():
                actions = getattr(m, "supported_actions", []) or []
                if any(a.lower() == "generatecontent" for a in actions if isinstance(a, str)):
                    name = getattr(m, "name", "")
                    # Some APIs return 'models/<id>'. Both forms are often accepted.
                    candidate = name.replace("models/", "") if isinstance(name, str) else None
                    for mid in filter(None, [candidate, name]):
                        try:
                            resp = client.models.generate_content(model=mid, contents=prompt)
                            return _first_available_text(resp)
                        except Exception as e:  # noqa: BLE001
                            last_err = e
                            continue
        except Exception:
            pass

        return (
            "Error en el análisis de sensibilidad: no se pudo generar contenido con ningún modelo. "
            f"Detalle: {last_err}"
        )
    except Exception:
        # Fall back to legacy SDK if the new one isn't installed.
        pass

    # Legacy SDK (deprecated): `pip install -U google-generativeai`
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        models_to_try = [model] if model else []
        models_to_try += [m for m in _MODEL_CANDIDATES if m != model]

        last_err = None
        for m in models_to_try:
            try:
                legacy_model = genai.GenerativeModel(m)
                resp = legacy_model.generate_content(prompt)
                # Legacy response usually has resp.text
                return _first_available_text(resp)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

        return (
            "Error en el análisis de sensibilidad: no se pudo generar contenido con ningún modelo (SDK legado). "
            f"Detalle: {last_err}"
        )
    except Exception as e:  # noqa: BLE001
        return (
            "Error en el análisis de sensibilidad: no se pudo inicializar el cliente de Gemini. "
            f"Detalle: {e}"
        )

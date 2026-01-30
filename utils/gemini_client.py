"""Cliente mínimo para Gemini (sin SDKs externos).

Objetivo del proyecto:
- **Cero librerías externas** para la parte matemática.
- La integración con IA es opcional y debe ser **no intrusiva**.

Este cliente usa SOLO librería estándar y hace una llamada HTTP al endpoint
oficial de Google Generative Language.

Config:
- Variable de entorno: GEMINI_API_KEY (recomendada).
- Archivo opcional en la raíz del proyecto: config.json con {"GEMINI_API_KEY": "..."}.
- Archivo de ejemplo: config.example.json (NO contiene clave real).

Notas:
- Si no hay API key, la app debe seguir funcionando con reportes determinísticos.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


# Modelos candidatos (se prueban en orden). El API puede variar por región/proyecto.
_DEFAULT_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Versiones del API a probar (primero la recomendada por docs).
_DEFAULT_VERSIONS: List[str] = ["v1beta", "v1"]


def _project_root() -> Path:
    # utils/ -> proyecto/
    return Path(__file__).resolve().parents[1]


def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Obtiene la API key desde argumento, env var o config.json."""
    if api_key and str(api_key).strip():
        return str(api_key).strip()

    env = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if env and env.strip():
        return env.strip()

    path = _project_root() / "config.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            key = data.get("GEMINI_API_KEY") or data.get("GOOGLE_API_KEY") or data.get("API_KEY")
            if key and str(key).strip():
                return str(key).strip()
        except Exception:
            return None

    return None


def _extract_text_from_response(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    out: List[str] = []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str) and t:
            out.append(t)
    return "".join(out).strip()


def generate_text(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """Genera texto con Gemini vía HTTP (sin SDK)."""

    key = get_api_key(api_key)
    if not key:
        raise RuntimeError("No se encontró API key (GEMINI_API_KEY).")

    prompt = str(prompt)
    models = [model] if model else list(_DEFAULT_MODELS)

    last_err: Optional[Exception] = None

    for m in models:
        if not m:
            continue
        m = str(m).strip()
        if m.startswith("models/"):
            m = m.split("/", 1)[1]

        for ver in _DEFAULT_VERSIONS:
            endpoint = (
                f"https://generativelanguage.googleapis.com/{ver}/models/"
                f"{urllib.parse.quote(m)}:generateContent"
            )

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": key,
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                txt = _extract_text_from_response(data)
                if txt:
                    return txt
            except urllib.error.HTTPError as e:
                last_err = e
                continue
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

    raise RuntimeError(
        "No se pudo generar texto con Gemini. "
        "Verifica tu API key, conexión a internet y el modelo configurado. "
        f"Último error: {last_err}"
    )

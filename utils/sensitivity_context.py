"""Almacén liviano de contexto para IA.

Permite que la pestaña "Sensibilidad" (global) consulte el último contexto
calculado por el programa en cualquiera de los módulos.

No usa librerías externas.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


_last_by_module: Dict[str, Dict[str, Any]] = {}
_last_module: Optional[str] = None


def set_last_facts(module: str, facts: Dict[str, Any]) -> None:
    """Guarda el último JSON de hechos para un módulo (pl/transporte/redes)."""
    global _last_module
    if not isinstance(module, str):
        return
    if not isinstance(facts, dict):
        return
    key = module.strip().lower()
    _last_by_module[key] = facts
    _last_module = key


def get_last_facts(preferred_module: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Obtiene el último JSON de hechos.

    Si preferred_module se indica, intenta devolver ese módulo.
    Si no existe, devuelve el último módulo registrado.
    """
    if isinstance(preferred_module, str) and preferred_module.strip():
        key = preferred_module.strip().lower()
        if key in _last_by_module:
            return _last_by_module[key]
    if _last_module and _last_module in _last_by_module:
        return _last_by_module[_last_module]
    return None


def get_last_module() -> Optional[str]:
    return _last_module

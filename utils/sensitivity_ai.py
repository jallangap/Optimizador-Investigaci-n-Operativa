"""Guardrails for AI sensitivity explanations (IO).

This project uses Gemini to *explain* sensitivity results. The numerical
results must come ONLY from the manual algorithms (Python puro).

To prevent conceptual mistakes / hallucinations, we:
1) Provide a strict JSON with facts (inputs + computed metrics).
2) Use a rule-heavy prompt with a fixed template.
3) Validate the response:
   - No new numbers.
   - No contradictions (e.g., says 'no es óptima' when is_optimal=True).
   - No forbidden concepts (e.g., 'costo de nodo').
4) Retry once with a stronger prompt.
5) Fallback to a deterministic report if it still fails.

Standard library only.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# NOTE:
# The UI imports and calls functions in this module directly (e.g., from
# views/SensibilidadView.py). Therefore, `generate_text` MUST exist at import time
# to avoid NameError. We import it from the project helper that abstracts Gemini
# SDK versions. If the import fails, we provide a safe fallback that returns an
# explanatory error message.
try:
    from utils.gemini_client import generate_text, get_api_key  # type: ignore
except Exception as _e:  # noqa: BLE001
    def generate_text(
        prompt: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        return (
            "Error en el análisis de sensibilidad: no se pudo importar el cliente de Gemini. "
            f"Detalle: {_e}"
        )

    def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
        return explicit if isinstance(explicit, str) and explicit.strip() else None


_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _offline_theory_answer(q: str) -> str:
    """Fallback teórico sin IA (para cuando no hay API key).

    Mantiene el proyecto funcional para quienes clonen el repo sin configurar Gemini.
    """
    nq = _norm(q)

    blocks: List[str] = []
    blocks.append("Modo sin IA: explicación teórica (no se usa Gemini porque no hay API key).")

    if "precio sombra" in nq or "shadow" in nq or ("dual" in nq and "precio" in nq):
        blocks.append(
            "\n**Precio sombra (variable dual y)**\n"
            "- Interpreta el cambio marginal del valor óptimo ante un cambio pequeño en el RHS (b) de una restricción.\n"
            "- Aproximación local: ΔZ ≈ yᵢ · Δbᵢ (mientras la base óptima no cambie).\n"
            "- Si la restricción está activa (slack=0), suele tener y≠0; si no está activa (slack>0), normalmente y≈0.\n"
            "- Importante: **no** es ‘impacto directo sobre x’, sino sobre el lado derecho de la restricción."
        )

    if "costo reducido" in nq or "coste reducido" in nq or "reduced cost" in nq:
        blocks.append(
            "\n**Costo reducido (rc)**\n"
            "- Para una variable NO básica, rc mide cuánto debe mejorar su coeficiente en la FO para que sea atractiva (entre a la base).\n"
            "- En PL (simplex), rc=0 para variables básicas en el óptimo.\n"
            "- En Transporte (MODI), el costo reducido de una celda NO básica: rcᵢⱼ = cᵢⱼ − (uᵢ + vⱼ).\n"
            "  En minimización: si todos los rc de NO básicas son ≥ 0, la solución es óptima."
        )

    if "holgura" in nq or "slack" in nq or "exceso" in nq or "surplus" in nq:
        blocks.append(
            "\n**Holgura / Exceso**\n"
            "- Holgura (≤): slack = b − Ax.\n"
            "- Exceso (≥): surplus = Ax − b.\n"
            "- slack=0 indica restricción activa (binding). slack>0 indica que no limita a la solución."
        )

    if any(k in nq for k in ["transporte", "modi", "vogel", "esquina", "costo total"]):
        blocks.append(
            "\n**Transporte: lectura de sensibilidad**\n"
            "- u (filas) y v (columnas) son potenciales; describen costos relativos.\n"
            "- Un rc negativo en una celda NO básica sugiere mejora posible (minimización): esa celda puede entrar al ciclo.\n"
            "- Cambios en oferta/demanda normalmente requieren re-balanceo y pueden cambiar la base (no es ‘+1 y ya’)."
        )

    if any(k in nq for k in ["redes", "arista", "capacidad", "flujo", "costo minimo", "costo mínimo"]):
        blocks.append(
            "\n**Redes: sensibilidad**\n"
            "- Se analiza cómo cambia la solución si varían **costos/capacidades de ARISTAS**.\n"
            "- Aristas saturadas (flow=cap) son candidatas a cuellos de botella.\n"
            "- No existe ‘costo de nodo’ en estos modelos (salvo formulaciones especiales no usadas aquí)."
        )

    blocks.append(
        "\nSi quieres activar IA en este mismo proyecto (sin tocar variables de entorno):\n"
        "1) Copia `config.example.json` a `config.json` (en la carpeta del proyecto, junto a `main.py`).\n"
        "2) Pega tu clave en `GEMINI_API_KEY`.\n"
        "(Por seguridad, `config.json` está en .gitignore.)"
    )

    return "\n".join(blocks)

# En la UI del proyecto, un reporte de sensibilidad debe ser breve y técnico.
# Limitamos tamaño para evitar repeticiones masivas del modelo.
_MAX_CHARS = 4500

# Encabezados obligatorios (el prompt también lo exige).
_SECTION_MARKERS = [
    "A)",
    "B)",
    "C)",
    "D)",
    "E)",
]


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def _norm(s: str) -> str:
    return _strip_accents(s).lower()


def _num_variants(x: float) -> Set[str]:
    """Return a set of string variants for a numeric value to allow formatting differences."""
    out: Set[str] = set()
    # Raw
    out.add(str(x))
    # Common compact formats
    for fmt in (".10g", ".8g", ".6g", ".12g"):
        try:
            out.add(format(x, fmt))
        except Exception:
            pass
    # If it's essentially an int, allow int string
    try:
        xi = int(round(float(x)))
        if abs(float(x) - xi) < 1e-9:
            out.add(str(xi))
    except Exception:
        pass
    return out


def _flatten_numbers(obj: Any) -> Set[str]:
    """Collect numeric token strings from a nested JSON-like structure."""
    nums: Set[str] = set()

    def rec(o: Any) -> None:
        if o is None:
            return
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            nums.update(_num_variants(float(o)))
            return
        if isinstance(o, str):
            # If the string contains numbers (e.g., constraint "2*x1+x2<=100"),
            # include them as allowed numeric tokens.
            for m in _NUM_RE.findall(o):
                nums.add(m)
            return
        if isinstance(o, dict):
            for v in o.values():
                rec(v)
            return
        if isinstance(o, (list, tuple, set)):
            for v in o:
                rec(v)
            return

    rec(obj)

    # Always allow these common tokens
    nums.update({"0", "1", "-1"})
    return nums


def _extract_numeric_tokens(text: str) -> Set[str]:
    return set(_NUM_RE.findall(text))


def _contains_any(text_norm: str, needles: Iterable[str]) -> bool:
    for n in needles:
        if n in text_norm:
            return True
    return False


def _validate_structure(text: str) -> List[str]:
    """Validaciones comunes: estructura y tamaño."""
    reasons: List[str] = []
    if not isinstance(text, str) or not text.strip():
        return ["La respuesta está vacía."]

    if len(text) > _MAX_CHARS:
        reasons.append("La respuesta es demasiado larga (posible repetición/alucinación).")

    # Exigir encabezados A)-E)
    idxs: List[int] = []
    for m in _SECTION_MARKERS:
        i = text.find(m)
        if i < 0:
            reasons.append(f"Falta el encabezado obligatorio '{m}'.")
        else:
            idxs.append(i)
    # Orden
    if len(idxs) == len(_SECTION_MARKERS) and any(idxs[i] >= idxs[i + 1] for i in range(len(idxs) - 1)):
        reasons.append("Los encabezados A)-E) no están en orden.")
    return reasons


def _require_values_in_text(text: str, values: List[float], label: str) -> List[str]:
    """Exige que la respuesta cite explícitamente ciertos valores numéricos reales."""
    reasons: List[str] = []
    tokens = _extract_numeric_tokens(text)
    missing: List[str] = []
    for v in values:
        try:
            vv = float(v)
        except Exception:
            continue
        ok = any(var in tokens for var in _num_variants(vv))
        if not ok:
            missing.append(str(vv))

    if missing:
        reasons.append(
            f"Falta citar valores numéricos reales del programa ({label}): "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return reasons


def _validate_pl(text: str, facts: Dict[str, Any]) -> List[str]:
    t = _norm(text)
    reasons: List[str] = []

    # Debe mencionar las tres piezas clásicas.
    if not _contains_any(t, ["holgura", "slack"]):
        reasons.append("PL: falta mencionar holguras/slacks (debe interpretarlas).")
    if not _contains_any(t, ["precio sombra", "shadow", "dual"]):
        reasons.append("PL: falta mencionar precios sombra (valores duales y).")
    if not _contains_any(t, ["costo reducido", "costos reducidos", "reduced"]):
        reasons.append("PL: falta mencionar costos reducidos (rc).")

    # Debe explicitar que y afecta al RHS (lado derecho b), no a x.
    if _contains_any(t, ["precio sombra", "shadow", "dual"]):
        if not _contains_any(t, ["rhs", "lado derecho", "b=", "b (", "b)", "termino independiente", "término independiente", "recursos", "r.h.s"]):
            reasons.append("PL: menciona precios sombra pero no aclara que aplican al RHS (lado derecho b).")

    # If facts include is_optimal, don't contradict
    if isinstance(facts.get("is_optimal"), bool):
        if facts["is_optimal"] and _contains_any(t, ["no es optima", "no es óptima", "no es optimo", "no es óptimo"]):
            reasons.append("PL: contradicción: el programa indica óptimo pero el texto dice que no.")

    # Exigir que cite valores reales: Z, x, slacks, y y rc (al menos para variables x).
    required_vals: List[float] = []
    if isinstance(facts.get("objective_value"), (int, float)):
        required_vals.append(float(facts["objective_value"]))

    x = facts.get("x")
    if isinstance(x, dict):
        for v in x.values():
            if isinstance(v, (int, float)):
                required_vals.append(float(v))

    constraints = facts.get("constraints")
    if isinstance(constraints, list):
        for c in constraints:
            if isinstance(c, dict):
                sv = c.get("slack")
                yv = c.get("shadow_price")
                if isinstance(sv, (int, float)):
                    required_vals.append(float(sv))
                if isinstance(yv, (int, float)):
                    required_vals.append(float(yv))

    rcs = facts.get("reduced_costs")
    if isinstance(rcs, dict):
        for k, v in rcs.items():
            if isinstance(k, str) and k.strip().lower().startswith("x") and isinstance(v, (int, float)):
                required_vals.append(float(v))

    reasons.extend(_require_values_in_text(text, required_vals, "Z/x/slacks/y/rc"))

    return reasons


def _validate_transporte(text: str, facts: Dict[str, Any]) -> List[str]:
    t = _norm(text)
    reasons: List[str] = []

    is_opt = facts.get("is_optimal")
    if isinstance(is_opt, bool) and is_opt:
        if _contains_any(t, ["no es optima", "no es óptima", "no es optimo", "no es óptimo"]):
            reasons.append("Transporte: contradicción: is_optimal=true pero el texto dice que no es óptima.")

        # También evitar frases típicas de no-optimalidad
        if _contains_any(t, ["no se ha encontrado", "no se encontro", "no se encontró"]):
            reasons.append("Transporte: contradicción: is_optimal=true pero el texto sugiere que no se encontró óptimo.")

    # Debe mencionar u y v (precios sombra de filas/columnas) y costos reducidos.
    if not _contains_any(t, ["u=", " u ", "potencial", "potenciales", "precios sombra"]):
        reasons.append("Transporte: falta interpretar los potenciales u (filas).")
    if not _contains_any(t, ["v=", " v ", "potencial", "potenciales", "precios sombra"]):
        reasons.append("Transporte: falta interpretar los potenciales v (columnas).")
    if not _contains_any(t, ["costo reducido", "costos reducidos", "reducido"]):
        reasons.append("Transporte: falta mencionar costos reducidos (MODI).")

    # "reduced costs negative in basic cells" is conceptually wrong in the optimal certificate
    if _contains_any(t, ["basica", "básica"]) and _contains_any(t, ["costo reducido", "costos reducidos", "reducido"]):
        if "neg" in t:
            reasons.append("Transporte: menciona costos reducidos negativos en celdas básicas (conceptualmente inválido).")

    # Exigir que cite valores reales: costo total, is_optimal, algunos u/v, y si hay, algún costo reducido.
    required_vals: List[float] = []
    if isinstance(facts.get("total_cost"), (int, float)):
        required_vals.append(float(facts["total_cost"]))

    # u/v (tomamos algunos para no exigir listas largas)
    u = facts.get("u")
    v = facts.get("v")
    if isinstance(u, list):
        for vv in u[: min(3, len(u))]:
            if isinstance(vv, (int, float)):
                required_vals.append(float(vv))
    if isinstance(v, list):
        for vv in v[: min(3, len(v))]:
            if isinstance(vv, (int, float)):
                required_vals.append(float(vv))

    # costos reducidos: tomar algunos valores (si existen)
    rc = facts.get("reduced_costs")
    if isinstance(rc, list):
        for item in rc[: min(3, len(rc))]:
            if isinstance(item, dict) and isinstance(item.get("rc"), (int, float)):
                required_vals.append(float(item["rc"]))

    reasons.extend(_require_values_in_text(text, required_vals, "costo/u/v/rc"))

    # No permitir que diga que el programa no calcula el costo cuando sí lo trae.
    if isinstance(facts.get("total_cost"), (int, float)) and _contains_any(t, ["no calcula", "no se calcula", "no calcula la funcion objetivo", "no calcula el costo"]):
        reasons.append("Transporte: afirma que el costo/FO no está calculado, pero total_cost está en el JSON.")

    return reasons


def _validate_redes(text: str, facts: Dict[str, Any]) -> List[str]:
    t = _norm(text)
    reasons: List[str] = []

    if _contains_any(
        t,
        [
            "costo de nodo",
            "costo del nodo",
            "coste de nodo",
            "costo en el nodo",
            "costo de los nodos",
            "costo de nodos",
            "costo en los nodos",
            "coste de los nodos",
            "coste de nodos",
        ],
    ):
        reasons.append("Redes: menciona 'costo de nodo', concepto no definido en el modelo.")

    # Debe hablar en términos de aristas/capacidades/pesos/costos.
    if not _contains_any(t, ["arista", "aristas", "edge", "edges"]):
        reasons.append("Redes: falta referenciar aristas (no uses 'costo de nodo').")

    # Bottleneck only if supported by saturated edges / min cut
    mentions_bottleneck = _contains_any(t, ["cuello de botella", "bottleneck", "arista critica", "aristas criticas", "aristas críticas"])
    sat = facts.get("saturated_edges")
    has_sat = isinstance(sat, list) and len(sat) > 0
    has_cut = bool(facts.get("min_cut"))
    if mentions_bottleneck and not (has_sat or has_cut):
        reasons.append("Redes: menciona cuellos de botella sin evidencia (no hay saturated_edges ni min_cut).")

    # Exigir que cite valores reales clave y, en flujo de costo mínimo, que cite al menos un detalle por arista.
    metodo = facts.get("metodo")
    required_vals: List[float] = []
    if metodo == "Flujo de Costo Mínimo":
        if isinstance(facts.get("flow_value"), (int, float)):
            required_vals.append(float(facts["flow_value"]))
        if isinstance(facts.get("total_cost"), (int, float)):
            required_vals.append(float(facts["total_cost"]))

        # Forzar al modelo a citar algún costo/capacidad/flujo de aristas.
        table = facts.get("edge_flow_table")
        if isinstance(table, list) and table:
            # Tomamos algunas cifras representativas distintas de (flow_value, total_cost)
            fv = float(facts.get("flow_value", 0) or 0)
            tc = float(facts.get("total_cost", 0) or 0)
            extra_flows: List[float] = []
            extra_costs: List[float] = []
            for row in table[: min(6, len(table))]:
                if not isinstance(row, dict):
                    continue
                f = row.get("flow")
                c = row.get("cost")
                cap = row.get("cap")
                if isinstance(f, (int, float)) and float(f) not in (fv,):
                    extra_flows.append(float(f))
                if isinstance(c, (int, float)) and float(c) not in (tc,):
                    extra_costs.append(float(c))
                if isinstance(cap, (int, float)) and float(cap) not in (fv, tc):
                    required_vals.append(float(cap))
            if extra_flows:
                required_vals.append(extra_flows[0])
            if extra_costs:
                required_vals.append(extra_costs[0])

    elif metodo == "Flujo Máximo":
        if isinstance(facts.get("flow_value"), (int, float)):
            required_vals.append(float(facts["flow_value"]))
        cut = facts.get("min_cut")
        if isinstance(cut, dict) and isinstance(cut.get("capacity"), (int, float)):
            required_vals.append(float(cut["capacity"]))

    elif metodo == "Ruta Más Corta":
        if isinstance(facts.get("distancia"), (int, float)):
            required_vals.append(float(facts["distancia"]))

    elif metodo == "Árbol de Mínima Expansión":
        if isinstance(facts.get("costo_total"), (int, float)):
            required_vals.append(float(facts["costo_total"]))

    if required_vals:
        reasons.extend(_require_values_in_text(text, required_vals, "valor/costo/aristas"))

    return reasons


def validate_response(module: str, facts: Dict[str, Any], text: str) -> Tuple[bool, List[str]]:
    """Validate AI response against facts."""
    reasons: List[str] = []

    # Estructura y tamaño
    reasons.extend(_validate_structure(text))

    allowed = _flatten_numbers(facts)
    tokens = _extract_numeric_tokens(text)
    extra = sorted([t for t in tokens if t not in allowed])
    if extra:
        # Some models output "+1"; token is "1" already. Still, keep strict.
        reasons.append(
            "Se detectaron números no presentes en los datos del programa (posible invención): "
            + ", ".join(extra[:12])
            + (" ..." if len(extra) > 12 else "")
        )

    if module == "pl":
        reasons.extend(_validate_pl(text, facts))
    elif module == "transporte":
        reasons.extend(_validate_transporte(text, facts))
    elif module == "redes":
        reasons.extend(_validate_redes(text, facts))

    return (len(reasons) == 0), reasons


def _prompt_header(module: str) -> str:
    base = (
        "Eres un asistente experto en Investigación Operativa (IO).\n"
        "Tu trabajo NO es calcular; el programa YA calculó los valores.\n"
        "Debes EXPLICAR usando exclusivamente los hechos del JSON.\n\n"
        "REGLAS ESTRICTAS (OBLIGATORIAS):\n"
        "1) PROHIBIDO inventar números, rangos, ejemplos hipotéticos o escenarios con cifras.\n"
        "2) PROHIBIDO contradecir el campo is_optimal / objective_value / total_cost / flow_value.\n"
        "3) Si un dato no existe en el JSON, responde exactamente: 'No calculado por el programa'.\n"
        "4) No introduzcas conceptos no definidos (p.ej., 'costo de nodo').\n"
    )

    if module == "pl":
        base += (
            "5) En PL: los precios sombra (y) se interpretan respecto al RHS (lado derecho b) de CADA restricción, "
            "no como efectos directos sobre x. Debes decirlo explícitamente.\n"
            "6) En PL: interpreta costos reducidos (rc) solo con base en los valores del JSON.\n"
            "7) En PL: DEBES citar números reales en C): (i) por cada restricción: slack y precio sombra y; "
            "(ii) por cada variable x: valor y costo reducido rc.\n"
        )
    elif module == "transporte":
        base += (
            "5) En Transporte (MODI): si is_optimal=true, debes declarar 'La solución es óptima'.\n"
            "6) En Transporte: costos reducidos negativos se evalúan en celdas NO básicas; no digas que son negativos en básicas.\n"
            "7) En Transporte: DEBES citar números reales en C): costo total, potenciales u y v, y costos reducidos (rc) de celdas NO básicas relevantes.\n"
        )
    elif module == "redes":
        base += (
            "5) En Redes: 'cuello de botella' solo si el JSON incluye saturated_edges o min_cut; "
            "en caso contrario, di 'No calculado por el programa'.\n"
            "6) En Redes: DEBES citar números reales y referenciar aristas: al menos un par de aristas con (capacidad, costo/peso y flujo o uso) según el método.\n"
        )

    base += "\nFORMATO DE SALIDA OBLIGATORIO (usa estos encabezados):\n"
    base += (
        "A) Resumen de hechos (del programa)\n"
        "B) Certificado / verificación (optimalidad / consistencia)\n"
        "C) Interpretación de sensibilidad (con teoría correcta)\n"
        "D) Qué pasaría si cambian parámetros (solo cualitativo si no hay rangos)\n"
        "E) Limitaciones (qué NO está calculado)\n"
    )
    return base


def build_prompt(module: str, facts: Dict[str, Any]) -> str:
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    return _prompt_header(module) + "\nHECHOS_JSON:\n" + facts_json + "\n"


def deterministic_report(module: str, facts: Dict[str, Any]) -> str:
    """Deterministic (non-AI) report, used as safe fallback."""

    if module == "pl":
        out = []
        out.append("A) Resumen de hechos (del programa)")
        out.append(f"- Objetivo: {facts.get('objective_sense')} | Valor óptimo: {facts.get('objective_value')}")
        out.append(f"- Variables: {facts.get('x')}")
        out.append("- Restricciones (slack, precio sombra y):")
        for r in facts.get("constraints", []):
            out.append(
                f"  * R{r.get('id')}: slack={r.get('slack')} | y={r.get('shadow_price')} | {r.get('expr')}"
            )
        out.append(f"- Costos reducidos: {facts.get('reduced_costs')}")
        out.append("\nB) Certificado / verificación")
        out.append(
            "- En un óptimo, para maximización se espera rc<=0 en variables no básicas (y rc=0 en básicas). "
            "Para minimización, rc>=0 en no básicas (dependiendo de la convención)."
        )
        out.append(f"- Estado reportado por el programa: is_optimal={facts.get('is_optimal')}")
        out.append("\nC) Interpretación de sensibilidad")
        out.append(
            "- Precios sombra (y): miden cómo cambia el valor óptimo ante un cambio marginal en el RHS (lado derecho b) "
            "de cada restricción, manteniendo la base. No son efectos directos sobre x." 
        )
        out.append(
            "- Holgura/exceso: indica si la restricción está activa (slack=0) o no (slack>0)."
        )
        out.append(
            "- Costo reducido (rc): indica el incentivo marginal de una variable no básica para entrar a la base." 
        )
        out.append("\nD) Qué pasaría si cambian parámetros")
        out.append("- Sin rangos de estabilidad calculados, solo puede afirmarse el efecto local usando y sobre RHS.")
        out.append("\nE) Limitaciones")
        out.append("- Rangos de aumento/disminución permitidos (allowable increase/decrease): No calculado por el programa.")
        return "\n".join(out)

    if module == "transporte":
        out = []
        out.append("A) Resumen de hechos (del programa)")
        out.append(f"- Método: {facts.get('metodo')} | Costo total: {facts.get('total_cost')}")
        out.append(f"- is_optimal: {facts.get('is_optimal')}")
        out.append("- u (filas) y v (columnas):")
        out.append(f"  u={facts.get('u')}\n  v={facts.get('v')}")
        out.append("- Celdas con costo reducido negativo (NO básicas):")
        out.append(str(facts.get("violations")))
        out.append("\nB) Certificado / verificación")
        out.append(
            "- En MODI: si todos los costos reducidos de celdas NO básicas son >= 0, la solución es óptima (minimización)."
        )
        out.append("\nC) Interpretación de sensibilidad")
        out.append(
            "- u y v son potenciales (precios sombra) asociados a filas/columnas; describen la estructura de costos relativos."
        )
        out.append("\nD) Qué pasaría si cambian parámetros")
        out.append(
            "- Cambios marginales en costos pueden evaluarse con costos reducidos; cambios en oferta/demanda requieren rebalancear." 
        )
        out.append("\nE) Limitaciones")
        out.append("- Rangos exactos de variación (sin cambiar base/ciclo): No calculado por el programa.")
        return "\n".join(out)

    # redes
    out = []
    out.append("A) Resumen de hechos (del programa)")
    out.append(f"- Método: {facts.get('metodo')}")
    if "flow_value" in facts:
        out.append(f"- Flujo total: {facts.get('flow_value')} | Costo total: {facts.get('total_cost')}")
    if facts.get("saturated_edges"):
        out.append(f"- Aristas saturadas (evidencia de cuello de botella): {facts.get('saturated_edges')}")
    out.append("\nB) Certificado / verificación")
    out.append("- El reporte se limita a hechos calculados; no se infieren cortes mínimos si no están provistos.")
    out.append("\nC) Interpretación de sensibilidad")
    out.append(
        "- Sensibilidad en redes se refiere a cómo cambian rutas/árbol/flujo si varían pesos, capacidades o costos de ARISTAS." 
        "No existe 'costo de nodo' en este modelo." 
    )
    out.append("\nD) Qué pasaría si cambian parámetros")
    out.append("- Si aumenta/disminuye el costo o capacidad de una arista usada/saturada, la solución puede cambiar.")
    out.append("\nE) Limitaciones")
    out.append("- Cortes mínimos / alternativas (k-shortest, edge sensitivity) : No calculado por el programa.")
    return "\n".join(out)


def generate_sensitivity_report(
    module: str,
    facts: Dict[str, Any],
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """Genera un reporte **técnicamente seguro** (A-E) y, opcionalmente, un anexo (F).

    - A-E: SIEMPRE determinístico (valores reales del motor matemático).
    - F: "Impacto en la toma de decisiones" usando Gemini **solo si hay API key**,
      con validación estricta y sin inventar números.

    Esto evita alucinaciones y mantiene el proyecto conforme a la regla de
    "cero librerías externas" para el cálculo.
    """

    # A-E: siempre con números reales calculados.
    base = deterministic_report(module, facts)

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return base

    # F: intento opcional. Si falla, simplemente no se agrega.
    try:
        f_text = generate_decision_commentary(module, facts, api_key=resolved_key, max_retries=max_retries)
    except Exception as e:  # noqa: BLE001
        return base + "\n\nF) Impacto en la toma de decisiones\n- IA no disponible: " + str(e)
    if f_text:
        return base + "\n\n" + f_text
    return base


def _build_decision_prompt(module: str, facts: Dict[str, Any]) -> str:
    """Prompt corto para el anexo F (sin introducir números nuevos)."""
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    return (
        "Eres un asistente de Investigación Operativa.\n"
        "Necesito un ANEXO 'F) Impacto en la toma de decisiones' basado SOLO en HECHOS_JSON.\n"
        "REGLAS OBLIGATORIAS:\n"
        "- NO inventes números, rangos, porcentajes ni ejemplos.\n"
        "- NO repitas A-E; aporta decisiones/impacto en lenguaje gerencial.\n"
        "- Si no hay datos para cuantificar, dilo explícitamente y manténlo cualitativo.\n"
        "- No contradigas is_optimal/valores.\n"
        "- Prohibido: 'costo de nodo', 'bottleneck por nodo', o cualquier concepto no estándar.\n"
        "FORMATO:\n"
        "F) Impacto en la toma de decisiones (IA)\n"
        "- 4 a 8 viñetas, claras y accionables.\n\n"
        f"MÓDULO={module}\n"
        "HECHOS_JSON:\n"
        + facts_json
        + "\n"
    )


def _validate_decision_commentary(module: str, facts: Dict[str, Any], text: str) -> Tuple[bool, List[str]]:
    """Validación específica para el anexo F (más estricta y cualitativa)."""
    reasons: List[str] = []
    if not text or not isinstance(text, str):
        return False, ["Respuesta vacía"]

    low = text.lower()
    if "f)" not in low:
        reasons.append("No incluye encabezado 'F)'.")

    # Evitar inventar números nuevos: usamos el validador general,
    # pero ignoramos numeración de listas ("1)", "2.") y etiquetas tipo R1/R2.
    text_for_numcheck = re.sub(r"(?m)^\s*\d+(?:\.|\))\s*", "", text)
    text_for_numcheck = re.sub(r"\bR\d+\b", "R", text_for_numcheck, flags=re.IGNORECASE)
    ok_numbers, r_numbers = _validate_no_new_numbers(facts, text_for_numcheck)
    if not ok_numbers:
        reasons.extend(["Introduce números no presentes en HECHOS_JSON."] + r_numbers)

    # No contradicciones básicas.
    is_opt = facts.get("is_optimal")
    if isinstance(is_opt, bool):
        if is_opt and ("no es óptima" in low or "no es optima" in low or "subópt" in low or "subopt" in low):
            reasons.append("Contradice is_optimal=True.")
        if (not is_opt) and ("es óptima" in low or "es optima" in low) and "no" not in low:
            reasons.append("Contradice is_optimal=False.")

    # Redes: prohibir 'costo de nodo'.
    if module.strip().lower() == "redes" and ("costo de nodo" in low or "coste de nodo" in low):
        reasons.append("Usa el concepto prohibido 'costo de nodo'.")

    return (len(reasons) == 0), reasons


def generate_decision_commentary(
    module: str,
    facts: Dict[str, Any],
    *,
    api_key: str,
    max_retries: int = 1,
) -> str:
    """Genera el anexo F con Gemini, si es posible.

    - Si la respuesta falla validación, reintenta con feedback.
    - Si vuelve a fallar, retorna "" para no contaminar el reporte.
    """
    prompt = _build_decision_prompt(module, facts)
    try:
        text = generate_text(prompt, api_key=api_key)
    except Exception:
        return ""
    ok, reasons = _validate_decision_commentary(module, facts, text)
    if ok:
        return text.strip()

    if max_retries <= 0:
        return ""

    retry_prompt = (
        prompt
        + "\n\nTU RESPUESTA ANTERIOR FUE RECHAZADA POR ESTOS MOTIVOS:\n- "
        + "\n- ".join(reasons[:10])
        + "\n\nReescribe cumpliendo estrictamente las reglas. "
        "No agregues números nuevos y manténlo cualitativo.\n"
    )
    try:
        text2 = generate_text(retry_prompt, api_key=api_key)
    except Exception:
        return ""

    ok2, _ = _validate_decision_commentary(module, facts, text2)
    return text2.strip() if ok2 else ""


def build_prompt_with_question(module: str, facts: Dict[str, Any], question: str) -> str:
    """Prompt para responder una pregunta del usuario usando hechos JSON.

    Se mantiene el mismo formato A-E para que la UI sea consistente.
    """
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    q = (question or "").strip()
    return (
        _prompt_header(module)
        + "\nPREGUNTA_DEL_USUARIO:\n"
        + q
        + "\n\nHECHOS_JSON:\n"
        + facts_json
        + "\n"
    )


def generate_contextual_answer(
    module: str,
    facts: Dict[str, Any],
    question: str,
    *,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """Responde una pregunta usando el contexto real calculado por el programa.

    - No inventa números.
    - Mantiene formato A-E.
    - Si la respuesta falla validación, se hace 1 reintento y luego fallback.
    """
    resolved_key = get_api_key(api_key)
    if not resolved_key:
        # Sin IA: contestamos con hechos y una guía corta.
        base = deterministic_report(module, facts)
        if question and question.strip():
            base += (
                "\n\nNota: no hay API key configurada para Gemini, así que la respuesta se limita "
                "a hechos calculados por el programa y teoría estándar.\n"
                "Si quieres activar IA dentro del proyecto, copia 'config.example.json' -> 'config.json' "
                "y pega GEMINI_API_KEY."
            )
        return base

    prompt = build_prompt_with_question(module, facts, question)
    for attempt in range(max_retries + 1):
        text = generate_text(prompt, api_key=resolved_key)
        ok, reasons = validate_response(module, facts, text)
        if ok:
            return text
        # Reintento con feedback técnico (sin introducir números nuevos)
        if attempt < max_retries:
            prompt = (
                build_prompt_with_question(module, facts, question)
                + "\n\nTU_RESPUESTA_FUE_RECHAZADA_POR_ESTAS_RAZONES:\n- "
                + "\n- ".join(reasons[:8])
                + "\n\nReescribe la respuesta corrigiendo exactamente esos puntos."
            )

    # Fallback seguro
    base = deterministic_report(module, facts)
    if question and question.strip():
        base += (
            "\n\nNota sobre tu pregunta: El programa no calcula por sí mismo decisiones de negocio (p.ej. 'rentable sí/no') "
            "si no se especifica qué restricción cambia y en cuánto.\n"
            "Con precios sombra (y), el cambio marginal en el óptimo se estima como: ΔZ ≈ yᵢ * Δbᵢ (manteniendo base).\n"
            "Si puedes mapear la pregunta a una restricción (RHS) concreta, se puede interpretar con esa fórmula."
        )
    return base


def generate_theory_answer(user_text: str, *, api_key: Optional[str] = None) -> str:
    """Respuesta teórica (sin usar hechos del programa).

    Útil para la pestaña global de Sensibilidad cuando el usuario no ha resuelto
    ningún problema aún o quiere una explicación conceptual.
    """
    q = (user_text or "").strip()
    if not q:
        return "Ingresa una pregunta o un caso para analizar."

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return _offline_theory_answer(q)

    prompt = (
        "Eres un experto en Investigación Operativa.\n"
        "Responde de forma clara y técnica, sin inventar cifras.\n"
        "Si el usuario provee números en su pregunta, puedes referirte a ellos, pero no agregues números nuevos.\n"
        "Si falta información para concluir, explica qué dato falta y da el procedimiento.\n\n"
        "PREGUNTA_DEL_USUARIO:\n"
        + q
        + "\n"
    )

    # Validación liviana: no agregar números que no estén en la pregunta.
    allowed = set(_extract_numeric_tokens(q))
    text = generate_text(prompt, api_key=resolved_key)
    extra = sorted([t for t in _extract_numeric_tokens(text) if t not in allowed])
    if extra:
        # Si el modelo inventa números, se devuelve una explicación sin números.
        return (
            "Puedo explicarlo conceptualmente, pero no puedo introducir cifras que no estén en tu pregunta.\n\n"
            "Guía general: identifica (i) función objetivo, (ii) restricciones (RHS), (iii) solución óptima y base.\n"
            "En PL, los precios sombra (y) miden el valor marginal de aumentar el RHS de una restricción; "
            "los costos reducidos (rc) miden cuánto debería mejorar el coeficiente de una variable no básica para entrar; "
            "las holguras indican si la restricción es activa.\n"
            "Si me das la formulación completa (o ejecutas el análisis en los módulos), puedo interpretarlo con los valores reales."
        )
    return text

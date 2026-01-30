"""Sensibilidad: reporte A–F (Python puro + IA opcional).

Reglas del proyecto:
- Los algoritmos de IO (PL / Transporte / Redes) se resuelven 100% "a mano".
- La IA SOLO redacta texto (interpretación / impacto decisional). NUNCA resuelve el problema.
- No existe interfaz tipo chat/Q&A: el flujo es
    1) Resolver el ejercicio
    2) "Analizar Sensibilidad" → reporte A–F

Este módulo:
- Genera A–E de forma determinística, usando SOLO los hechos numéricos calculados.
- Genera F (impacto en decisiones) con IA si hay API key; si no, degrada a una versión offline.
- Valida que la salida IA NO invente números ni agregue secciones extra.

Dependencias: solo librería estándar + (opcional) llamada HTTP a Gemini vía utils/gemini_client.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    # Cliente HTTP (sin SDK) provisto por el proyecto.
    from utils.gemini_client import generate_text, get_api_key as _get_api_key  # type: ignore
except Exception:  # noqa: BLE001
    generate_text = None  # type: ignore

    def _get_api_key(_explicit: Optional[str] = None) -> Optional[str]:
        return None


# -----------------------------
# Utilidades de validación
# -----------------------------

_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    return _NUM_RE.findall(text)


def _num_variants(x: float) -> Set[str]:
    """Representaciones típicas para no penalizar formato (1 vs 1.0 vs 1.00)."""
    out: Set[str] = set()
    try:
        xf = float(x)
    except Exception:
        return out

    # Formatos comunes
    out.add(str(int(round(xf))) if abs(xf - round(xf)) < 1e-12 else "")
    out.add(f"{xf}")
    out.add(f"{xf:.6g}")
    out.add(f"{xf:.12g}")

    # Versiones sin ceros finales
    for s in list(out):
        if not s:
            continue
        if "." in s:
            out.add(s.rstrip("0").rstrip("."))

    # Limpiar vacíos
    out.discard("")
    return out


def _allowed_numbers_from_facts(facts: Dict[str, Any]) -> Set[str]:
    """Construye el conjunto de tokens numéricos permitidos a partir del JSON de hechos."""
    raw = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    toks = set(_extract_numeric_tokens(raw))

    allowed: Set[str] = set()
    for t in toks:
        allowed.add(t)
        try:
            allowed |= _num_variants(float(t))
        except Exception:
            continue

    # Regla del proyecto: se permite mencionar "una unidad" de cambio marginal,
    # pero evitamos introducir el token "1" si no está.
    # Aun así, dejamos el dígito 1 permitido para evitar falsos positivos si la IA lo usa.
    allowed.add("1")
    allowed.add("1.0")
    return allowed


def _strip_list_numbering(text: str) -> str:
    """Elimina numeración típica de listas para que no cuente como 'números nuevos'."""
    if not text:
        return text
    lines: List[str] = []
    for ln in text.splitlines():
        ln2 = re.sub(r"^\s*(?:[-*•]+\s+)", "", ln)
        ln2 = re.sub(r"^\s*\(?\d+\)?[\).:-]\s+", "", ln2)
        lines.append(ln2)
    return "\n".join(lines)


def _strip_label_numbers(text: str) -> str:
    """Normaliza etiquetas tipo R1/x2/celda(1,2) para que no gatillen validación."""
    if not text:
        return text
    # R1 -> R, x2 -> x
    text = re.sub(r"\bR\s*\d+\b", "R", text)
    text = re.sub(r"\bx\s*\d+\b", "x", text, flags=re.IGNORECASE)
    # (i,j) -> (i,j) sin tocar (porque normalmente viene en hechos); pero evitamos 'celda 1,2'
    text = re.sub(r"\bcelda\s*\(\s*\d+\s*,\s*\d+\s*\)", "celda(i,j)", text, flags=re.IGNORECASE)
    return text


def _validate_no_new_numbers(facts: Dict[str, Any], text: str) -> Tuple[bool, List[str]]:
    allowed = _allowed_numbers_from_facts(facts)
    cleaned = _strip_label_numbers(_strip_list_numbering(text))
    extra = sorted({t for t in _extract_numeric_tokens(cleaned) if t not in allowed})
    if extra:
        return False, [f"La IA introdujo números no presentes en los hechos: {extra}"]
    return True, []


# -----------------------------
# Reporte determinístico (A–E)
# -----------------------------


def _module_key(module: Optional[str], facts: Dict[str, Any]) -> str:
    m = (module or "").strip().lower()
    if not m:
        m = str(facts.get("module") or "").strip().lower()
    if m in ("pl", "programacion lineal", "programación lineal"):
        return "pl"
    if m in ("transporte", "transport"):
        return "transporte"
    if m in ("redes", "red", "network"):
        return "redes"
    return m or ""


def deterministic_report(module: str, facts: Dict[str, Any]) -> str:
    """Secciones A–E: SOLO hechos del programa + teoría sin inventar cifras."""
    m = _module_key(module, facts)

    if m == "pl":
        out: List[str] = []
        out.append("A) Resumen de hechos (solo datos calculados por el programa)")
        out.append(f"- Objetivo: {facts.get('objective_sense')} | Valor óptimo: {facts.get('objective_value')}")
        out.append(f"- Variables: {facts.get('x')}")
        out.append("- Restricciones (holgura y precio sombra):")
        for r in facts.get("constraints", []) or []:
            out.append(
                f"  * R{r.get('id')}: slack={r.get('slack')} | y={r.get('shadow_price')} | {r.get('expr')}"
            )
        out.append(f"- Costos reducidos: {facts.get('reduced_costs')}")

        out.append("\nB) Certificado / verificación")
        out.append(f"- Consistencia reportada por el programa: is_optimal={facts.get('is_optimal')}")
        out.append(
            "- Convención: en maximización, en una solución óptima los costos reducidos de variables no básicas "
            "no deberían ser positivos; en minimización, no deberían ser negativos. "
            "(El programa verifica esto con su base final)."
        )

        out.append("\nC) Interpretación teórica correcta (sin inventar números)")
        out.append(
            "- Precio sombra (y): cambio marginal del valor óptimo ante cambios pequeños en el RHS de una restricción, "
            "válido localmente mientras la base no cambie."
        )
        out.append("- Holgura: indica si una restricción está activa o si hay margen (capacidad ociosa).")
        out.append(
            "- Costo reducido: mide qué tan 'lejos' está una variable no básica de volverse atractiva para entrar a la base, "
            "bajo la convención del sentido del problema."
        )

        out.append("\nD) Qué pasaría si cambian parámetros")
        out.append(
            "- Si cambia un RHS: el efecto marginal se interpreta con y (siempre de forma local). "
            "Si cambian coeficientes de la FO o de A de forma relevante, debe re-optimizarse."
        )

        out.append("\nE) Limitaciones (lo que NO se calcula)")
        out.append("- Rangos de estabilidad (allowable increase/decrease) para RHS y coeficientes: No calculado por el programa.")
        out.append("- Casos con cambios grandes o cambio de base: requieren re-optimización.")
        return "\n".join(out)

    if m == "transporte":
        out = []
        out.append("A) Resumen de hechos (solo datos calculados por el programa)")
        out.append(f"- Método: {facts.get('metodo')} | Costo total: {facts.get('total_cost')}")
        out.append(f"- is_optimal (MODI): {facts.get('is_optimal')}")
        out.append(f"- u (filas): {facts.get('u')}")
        out.append(f"- v (columnas): {facts.get('v')}")
        out.append("- Celdas NO básicas con costo reducido negativo (si existen):")
        out.append(str(facts.get("violations") or []))

        out.append("\nB) Certificado / verificación")
        out.append(
            "- Prueba MODI (minimización): la solución es óptima si todas las celdas NO básicas tienen costo reducido no negativo."
        )

        out.append("\nC) Interpretación teórica correcta (sin inventar números)")
        out.append(
            "- Los potenciales u y v actúan como precios implícitos de oferta/demanda para explicar costos relativos. "
            "El costo reducido de una celda mide la oportunidad de mejorar el costo total si esa ruta entra en la base."
        )

        out.append("\nD) Qué pasaría si cambian parámetros")
        out.append(
            "- Si cambian costos unitarios, el patrón de envíos puede cambiar; si cambian oferta/demanda, el problema debe rebalancearse "
            "y re-optimizarse."
        )

        out.append("\nE) Limitaciones (lo que NO se calcula)")
        out.append("- Rangos de estabilidad sin cambiar el ciclo base: No calculado por el programa.")
        out.append("- Interpretaciones monetarias (ahorro exacto) requieren recomputar el plan: el reporte no estima beneficios.")
        return "\n".join(out)

    # redes
    out = []
    out.append("A) Resumen de hechos (solo datos calculados por el programa)")
    out.append(f"- Método: {facts.get('metodo')}")

    if facts.get("metodo") in ("Ruta Más Corta", "Árbol de Mínima Expansión"):
        if "ruta" in facts:
            out.append(f"- Ruta: {facts.get('ruta')} | Distancia: {facts.get('distancia')}")
            if facts.get("used_edges"):
                out.append(f"- Aristas usadas (con pesos): {facts.get('used_edges')}")
        if "arbol" in facts:
            out.append(f"- Árbol: {facts.get('arbol')} | Costo total: {facts.get('costo_total')}")

    if "flow_value" in facts:
        out.append(f"- Flujo total: {facts.get('flow_value')}")
    if "total_cost" in facts:
        out.append(f"- Costo total: {facts.get('total_cost')}")
    if facts.get("saturated_edges"):
        out.append(f"- Aristas saturadas: {facts.get('saturated_edges')}")
    if facts.get("min_cut"):
        out.append(f"- Corte mínimo (si calculado): {facts.get('min_cut')}")
    if "is_optimal" in facts:
        out.append(f"- is_optimal: {facts.get('is_optimal')}")

    out.append("\nB) Certificado / verificación")
    if facts.get("metodo") == "Flujo Máximo" and facts.get("min_cut"):
        out.append("- Certificado: el programa compara el valor del flujo con la capacidad del corte mínimo calculado.")
    else:
        out.append("- Certificado completo depende del método; se reporta solo lo que el programa calculó.")

    out.append("\nC) Interpretación teórica correcta (sin inventar números)")
    out.append(
        "- Sensibilidad en redes se analiza sobre ARISTAS (peso/costo/capacidad). "
        "Cambiar parámetros de aristas usadas o saturadas puede cambiar rutas, árbol o flujo." 
    )

    out.append("\nD) Qué pasaría si cambian parámetros")
    out.append(
        "- Si sube el peso/costo de una arista usada, pueden aparecer alternativas; si baja la capacidad de una arista crítica, puede reducir el flujo o forzar desvíos. "
        "Para cuantificarlo, se requiere re-ejecutar el algoritmo."
    )

    out.append("\nE) Limitaciones (lo que NO se calcula)")
    out.append("- Sensibilidad exacta por arista (rangos de pesos/costos/capacidades que preservan la solución): No calculado por el programa.")
    out.append("- Alternativas avanzadas (k-rutas, análisis exhaustivo de cortes): No calculado por el programa.")

    return "\n".join(out)


# -----------------------------
# Anexo decisional (F) – IA opcional
# -----------------------------


def _human_refs(module_key: str, facts: Dict[str, Any]) -> str:
    """Referencias humanas para anclar la IA sin exponer claves JSON."""
    if module_key == "pl":
        lines = []
        for r in facts.get("constraints", []) or []:
            rid = r.get("id")
            lines.append(
                f"- R{rid}: slack={r.get('slack')} | y={r.get('shadow_price')} | {r.get('expr')}"
            )
        return "REFERENCIAS_RESTRICCIONES:\n" + "\n".join(lines)

    if module_key == "transporte":
        lines = []
        for v in facts.get("violations", []) or []:
            cell = v.get("cell")
            lines.append(f"- celda{tuple(cell) if isinstance(cell, list) else cell}: rc={v.get('rc')} | c={v.get('c')}")
        return "REFERENCIAS_CELDAS:\n" + ("\n".join(lines) if lines else "- (sin celdas rc<0 reportadas)")

    if module_key == "redes":
        lines = []
        for e in facts.get("saturated_edges", []) or []:
            lines.append(f"- arista {e.get('u')}->{e.get('v')}: flow={e.get('flow')} | cap={e.get('cap')}")
        return "REFERENCIAS_ARISTAS:\n" + ("\n".join(lines) if lines else "- (sin aristas saturadas reportadas)")

    return "REFERENCIAS:\n- (módulo no reconocido)"


def _facts_for_ai(module_key: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce el JSON a lo esencial para que la IA no divague."""
    if module_key == "pl":
        return {
            "objective_sense": facts.get("objective_sense"),
            "objective_value": facts.get("objective_value"),
            "is_optimal": facts.get("is_optimal"),
            "x": facts.get("x"),
            "reduced_costs": facts.get("reduced_costs"),
            "constraints": [
                {
                    "id": r.get("id"),
                    "expr": r.get("expr"),
                    "slack": r.get("slack"),
                    "shadow_price": r.get("shadow_price"),
                }
                for r in (facts.get("constraints") or [])
            ],
        }

    if module_key == "transporte":
        return {
            "metodo": facts.get("metodo"),
            "total_cost": facts.get("total_cost"),
            "is_optimal": facts.get("is_optimal"),
            "u": facts.get("u"),
            "v": facts.get("v"),
            "violations": facts.get("violations") or [],
        }

    if module_key == "redes":
        return {
            "metodo": facts.get("metodo"),
            "flow_value": facts.get("flow_value"),
            "total_cost": facts.get("total_cost"),
            "is_optimal": facts.get("is_optimal"),
            "saturated_edges": facts.get("saturated_edges") or [],
            "min_cut": facts.get("min_cut"),
            "ruta": facts.get("ruta"),
            "distancia": facts.get("distancia"),
            "arbol": facts.get("arbol"),
            "costo_total": facts.get("costo_total"),
        }

    # fallback: todo (pero igual validaremos números)
    return facts


def _build_decision_prompt(module_key: str, facts: Dict[str, Any]) -> str:
    facts_ai = _facts_for_ai(module_key, facts)
    facts_json = json.dumps(facts_ai, ensure_ascii=False, indent=2)
    refs = _human_refs(module_key, facts)

    module_rules = ""
    if module_key == "pl":
        module_rules = (
            "REGLAS (PL):\n"
            "- Conecta siempre un HECHO (y / slack / rc) con una decisión (capacidad, priorización de recursos, re-optimizar).\n"
            "- Reconoce validez local: precios sombra y costos reducidos son interpretables mientras no cambie la base.\n"
        )
    elif module_key == "transporte":
        module_rules = (
            "REGLAS (Transporte):\n"
            "- MODI (minimización): si existen celdas NO básicas con rc negativo, hay oportunidad de mejora; si no, es consistente con óptimo.\n"
            "- Cambios de oferta/demanda o costos requieren re-balanceo y re-optimización.\n"
        )
    elif module_key == "redes":
        module_rules = (
            "REGLAS (Redes):\n"
            "- Sensibilidad se analiza sobre ARISTAS (peso/costo/capacidad).\n"
            "- Solo hables de cuellos de botella si hay aristas saturadas o un certificado (min_cut) en los hechos.\n"
        )

    return (
        "Eres un asistente experto en Investigación Operativa.\n"
        "Tu trabajo NO es calcular soluciones; el programa ya calculó todo.\n"
        "Debes redactar SOLO la sección F) con enfoque decisional.\n\n"
        "REGLAS ESTRICTAS (OBLIGATORIAS):\n"
        "1) PROHIBIDO inventar números o rangos.\n"
        "2) Puedes usar ÚNICAMENTE números presentes en HECHOS_JSON (o decir 'No calculado por el programa').\n"
        "3) NO hagas preguntas al usuario, NO agregues secciones extra.\n"
        "4) No cites claves técnicas del JSON. Usa etiquetas humanas (R1, celda(i,j), arista u->v).\n\n"
        + module_rules
        + "\nFORMATO:\n"
        "F) Impacto en la toma de decisiones (IA)\n"
        "- Viñetas concretas que citen HECHOS reales del programa y expliquen implicación práctica.\n"
        "- Incluye: (i) priorización, (ii) verificación/consistencia con optimalidad, (iii) validez local/riesgo de re-optimizar.\n"
        "- NO repitas A)–E).\n\n"
        + refs
        + "\n\nHECHOS_JSON:\n"
        + facts_json
        + "\n"
    )


def deterministic_decision_addendum(module: str, facts: Dict[str, Any]) -> str:
    """Fallback offline para F) (sin IA), aún útil y basado en hechos."""
    m = _module_key(module, facts)
    lines: List[str] = []
    lines.append("F) Impacto en la toma de decisiones (IA)")

    if m == "pl":
        cons = facts.get("constraints", []) or []
        # Prioridad: restricciones activas (slack≈0) con |y| más alto
        active = []
        for r in cons:
            try:
                slack = float(r.get("slack"))
            except Exception:
                slack = None
            if slack is not None and abs(slack) < 1e-9:
                active.append(r)

        if active:
            # ordenar por |y|
            def _yabs(rr: Dict[str, Any]) -> float:
                try:
                    return abs(float(rr.get("shadow_price")))
                except Exception:
                    return -1.0

            active_sorted = sorted(active, key=_yabs, reverse=True)
            top = active_sorted[0]
            lines.append(
                f"- Priorización: la restricción R{top.get('id')} aparece activa (slack={top.get('slack')}) y tiene y={top.get('shadow_price')}; "
                "si puedes ampliar un recurso, empieza por la restricción con mayor impacto marginal (y en valor absoluto)."
            )
            if len(active_sorted) > 1:
                sec = active_sorted[1]
                lines.append(
                    f"- Comparación: otra restricción activa es R{sec.get('id')} con y={sec.get('shadow_price')}; "
                    "si el presupuesto es limitado, compara estos y para decidir dónde invertir primero."
                )
        else:
            lines.append(
                "- Priorización: el programa no identifica restricciones activas con holgura cercana a cero; "
                "con los datos actuales no se observa un 'cuello' claro por holgura." 
            )

        lines.append(
            f"- Verificación: el programa reporta is_optimal={facts.get('is_optimal')}; "
            "si cambia este estado, re-ejecuta el método antes de tomar decisiones." 
        )

        # Costos reducidos: resaltar variables con rc más extremos (sin inventar)
        rc = facts.get("reduced_costs") or {}
        # Filtrar solo x* si existe
        rc_x = {k: v for k, v in rc.items() if isinstance(k, str) and k.strip().lower().startswith('x')}
        if rc_x:
            # ordenar por magnitud
            items = []
            for k, v in rc_x.items():
                try:
                    items.append((k, float(v)))
                except Exception:
                    continue
            items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            if items:
                k0, v0 = items[0]
                lines.append(
                    f"- Sensibilidad por variable: {k0} tiene costo reducido rc={v0}. "
                    "En decisiones de portafolio/producto, variables con rc desfavorable requieren mejorar su coeficiente en la FO "
                    "para volverse competitivas (según la convención del sentido)."
                )

        lines.append(
            "- Riesgo/validez local: precios sombra y costos reducidos describen efectos marginales locales; "
            "si hay cambios grandes en RHS o coeficientes, debes re-optimizar."
        )
        lines.append(
            "- Limitación decisional: sin rangos de estabilidad, no se puede asegurar hasta qué punto puedes cambiar parámetros sin cambiar la base." 
        )

    elif m == "transporte":
        lines.append(
            f"- Verificación (MODI): el programa reporta is_optimal={facts.get('is_optimal')}. "
            "En minimización, rc negativo en una celda NO básica indica oportunidad de reducir costo." 
        )

        viol = facts.get("violations") or []
        if viol:
            v0 = viol[0]
            lines.append(
                f"- Priorización: existe al menos una celda NO básica con rc negativo (por ejemplo celda={v0.get('cell')} con rc={v0.get('rc')}). "
                "Esa es candidata natural para entrar al ciclo de mejora y reducir el costo total, si las restricciones lo permiten."
            )
        else:
            lines.append(
                "- Priorización: no se reportan celdas NO básicas con rc negativo; el plan es consistente con óptimo bajo MODI." 
            )

        lines.append(
            f"- Costo total actual: {facts.get('total_cost')}. "
            "Si se renegocian costos unitarios o cambia oferta/demanda, el costo total puede variar y se debe re-optimizar." 
        )
        lines.append(
            f"- Potenciales u y v: u={facts.get('u')} y v={facts.get('v')} resumen la estructura de costos relativos; "
            "sirven para explicar por qué ciertas rutas son más/menos atractivas sin recalcular todo." 
        )
        lines.append(
            "- Riesgo: cambios en oferta/demanda requieren rebalanceo (posible fila/columna dummy) y recomputar el plan." 
        )
        lines.append(
            "- Limitación decisional: el reporte no calcula ahorros exactos por cambio marginal; para cuantificar, hay que re-ejecutar." 
        )

    elif m == "redes":
        lines.append(
            f"- Verificación: método={facts.get('metodo')}. Si el programa reporta is_optimal={facts.get('is_optimal')}, "
            "úsalo solo como certificado cuando esté soportado (por ejemplo, flujo máximo con corte mínimo)."
        )

        sat = facts.get("saturated_edges") or []
        if sat:
            e0 = sat[0]
            lines.append(
                f"- Priorización: hay aristas saturadas (p.ej. {e0.get('u')}->{e0.get('v')} con flow={e0.get('flow')} y cap={e0.get('cap')}). "
                "Si buscas aumentar desempeño, una opción es ampliar capacidad o crear alternativa para desahogar esas aristas." 
            )
        else:
            lines.append(
                "- Priorización: no se reportan aristas saturadas; no hay evidencia de cuello de botella por saturación en los hechos." 
            )

        if facts.get("min_cut"):
            lines.append(
                f"- Certificado: se calculó min_cut={facts.get('min_cut')}. "
                "Cambios en aristas del corte pueden afectar el flujo máximo y son puntos naturales de intervención." 
            )

        if "flow_value" in facts:
            lines.append(
                f"- Métrica de salida: flujo_total={facts.get('flow_value')}. Cambios en capacidades/costos de aristas usadas pueden alterar este valor al re-optimizar." 
            )
        if "total_cost" in facts:
            lines.append(
                f"- Métrica de salida: costo_total={facts.get('total_cost')}. Si se cambian costos unitarios de aristas usadas, el costo mínimo puede variar y se debe recalcular." 
            )

        lines.append(
            "- Riesgo/validez: la sensibilidad en redes es altamente dependiente de la estructura; incluso cambios pequeños pueden cambiar rutas/árbol/flujo. "
            "Para cuantificar, re-ejecuta el método con los nuevos parámetros."
        )

    else:
        lines.append("- Módulo no reconocido para anexo decisional.")

    return "\n".join(lines).strip()


def _validate_decision_commentary(module: str, facts: Dict[str, Any], text: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not isinstance(text, str) or not text.strip():
        return False, ["La respuesta está vacía."]

    low = _norm(text)
    if "f)" not in low:
        reasons.append("No incluye el encabezado 'F)'.")

    # Prohibido agregar secciones extra (A–E, G, etc.)
    for bad in ("a)", "b)", "c)", "d)", "e)", "g)"):
        if bad in low:
            reasons.append("Agrega secciones extra fuera de F).")
            break

    # Prohibido hacer preguntas
    if "?" in text or "pregunta" in low or "siguientes pasos" in low:
        reasons.append("Incluye preguntas o apartado tipo Q&A.")

    # Prohibido citar claves técnicas
    forbidden = ["constraints[", "reduced_costs[", "hechos_json", "json", "id:"]
    if any(k in low for k in forbidden):
        reasons.append("Cita claves técnicas; debe usar etiquetas humanas (R1, celda(i,j), arista u->v).")

    ok_nums, nums_reasons = _validate_no_new_numbers(facts, text)
    if not ok_nums:
        reasons.extend(nums_reasons)

    # Consistencia básica con is_optimal (si existe)
    try:
        is_opt = facts.get("is_optimal")
        if isinstance(is_opt, bool):
            if is_opt and ("no es optima" in low or "no es óptima" in low):
                reasons.append("Contradicción: el programa reporta is_optimal=True.")
            if (is_opt is False) and ("es optima" in low or "es óptima" in low):
                reasons.append("Contradicción: el programa reporta is_optimal=False.")
    except Exception:
        pass

    # Redes: prohibir 'costo de nodo'
    if _module_key(module, facts) == "redes" and "costo de nodo" in low:
        reasons.append("Concepto inválido: 'costo de nodo' no aplica; use costos de aristas.")

    return (len(reasons) == 0), reasons


def generate_decision_commentary(
    module: str,
    facts: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """Genera F) con IA (si hay API key) o fallback offline determinístico."""

    # Si el cliente IA no está disponible, degradar.
    if generate_text is None:
        return deterministic_decision_addendum(module, facts)

    key = _get_api_key(api_key)
    if not key:
        return deterministic_decision_addendum(module, facts)

    mk = _module_key(module, facts)
    prompt = _build_decision_prompt(mk, facts)

    # Intento + reintentos controlados
    tries = max(0, int(max_retries))
    for attempt in range(tries + 1):
        try:
            text_out = generate_text(prompt, api_key=key)
        except Exception:
            text_out = ""

        ok, _reasons = _validate_decision_commentary(module, facts, text_out)
        if ok:
            return text_out.strip()

        # reforzar constraints (sin introducir números)
        prompt = (
            prompt
            + "\n\nIMPORTANTE: tu respuesta anterior fue inválida. "
            "Devuelve SOLO la sección F) y no inventes cifras ni rangos. "
            "No hagas preguntas."
        )

    # Si no pasa validación, fallback offline
    return deterministic_decision_addendum(module, facts)


# -----------------------------
# Ensamblador final A–F
# -----------------------------


def generate_sensitivity_report(
    module: str,
    facts: Dict[str, Any],
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """Genera el reporte completo A–F.

    - A–E: determinístico (solo hechos + teoría).
    - F: IA opcional con validación estricta; degrada a offline si no hay key o falla.
    """

    base = deterministic_report(module, facts)

    # Siempre incluir F (IA o fallback offline)
    f_text = generate_decision_commentary(module, facts, api_key=api_key, max_retries=max_retries)
    if f_text:
        return base + "\n\n" + f_text.strip()
    return base

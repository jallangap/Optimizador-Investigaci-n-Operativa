"""Sensibilidad: reporte A–F (Python puro + IA opcional).

Reglas del proyecto:
- Los algoritmos de IO (PL / Transporte / Redes) se resuelven 100% "a mano".
- La IA SOLO redacta texto (interpretación / impacto decisional). NUNCA resuelve el problema.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    # Cliente HTTP (sin SDK) provisto por el proyecto.
    from utils.gemini_client import generate_text, get_api_key as _get_api_key
except Exception:
    generate_text = None

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

    out.add(str(int(round(xf))) if abs(xf - round(xf)) < 1e-12 else "")
    out.add(f"{xf}")
    out.add(f"{xf:.6g}")
    out.add(f"{xf:.12g}")

    for s in list(out):
        if not s:
            continue
        if "." in s:
            out.add(s.rstrip("0").rstrip("."))
    out.discard("")
    return out


def _allowed_numbers_from_facts(facts: Dict[str, Any], context_text: str = "") -> Set[str]:
    """Construye el conjunto de tokens numéricos permitidos a partir de hechos y contexto."""
    raw_facts = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    # Combinamos texto de hechos y el contexto del usuario para extraer números válidos
    combined_text = raw_facts + " " + context_text
    
    toks = set(_extract_numeric_tokens(combined_text))

    allowed: Set[str] = set()
    for t in toks:
        allowed.add(t)
        try:
            allowed |= _num_variants(float(t))
        except Exception:
            continue

    allowed.add("1")
    allowed.add("1.0")
    return allowed


def _strip_list_numbering(text: str) -> str:
    if not text:
        return text
    lines: List[str] = []
    for ln in text.splitlines():
        ln2 = re.sub(r"^\s*(?:[-*•]+\s+)", "", ln)
        ln2 = re.sub(r"^\s*\(?\d+\)?[\).:-]\s+", "", ln2)
        lines.append(ln2)
    return "\n".join(lines)


def _strip_label_numbers(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\bR\s*\d+\b", "R", text)
    text = re.sub(r"\bx\s*\d+\b", "x", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcelda\s*\(\s*\d+\s*,\s*\d+\s*\)", "celda(i,j)", text, flags=re.IGNORECASE)
    return text


def _validate_no_new_numbers(facts: Dict[str, Any], text: str, context_text: str = "") -> Tuple[bool, List[str]]:
    allowed = _allowed_numbers_from_facts(facts, context_text)
    cleaned = _strip_label_numbers(_strip_list_numbering(text))
    extra = sorted({t for t in _extract_numeric_tokens(cleaned) if t not in allowed})
    if extra:
        return False, [f"La IA introdujo números no presentes en los hechos ni el contexto: {extra}"]
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
    out: List[str] = []

    if m == "pl":
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
        out.append("- Convención: Verifica costos reducidos según maximización/minimización.")

        out.append("\nC) Interpretación teórica correcta")
        out.append("- Precio sombra (y): cambio marginal del valor óptimo ante cambios en el RHS (válido localmente).")
        out.append("- Holgura: indica si una restricción está activa o hay capacidad ociosa.")

        out.append("\nD) Qué pasaría si cambian parámetros")
        out.append("- Si cambia un RHS: el efecto marginal se interpreta con y. Si hay cambios grandes, re-optimizar.")

        out.append("\nE) Limitaciones")
        out.append("- No se calculan rangos de estabilidad.")
        return "\n".join(out)

    if m == "transporte":
        out.append("A) Resumen de hechos")
        out.append(f"- Método: {facts.get('metodo')} | Costo total: {facts.get('total_cost')}")
        out.append(f"- is_optimal (MODI): {facts.get('is_optimal')}")
        return "\n".join(out)

    # redes y default
    out.append("A) Resumen de hechos")
    out.append(f"- Método: {facts.get('metodo')}")
    if "flow_value" in facts:
        out.append(f"- Flujo total: {facts.get('flow_value')}")
    return "\n".join(out)


# -----------------------------
# Anexo decisional (F) – IA opcional
# -----------------------------


def _human_refs(module_key: str, facts: Dict[str, Any]) -> str:
    if module_key == "pl":
        lines = []
        for r in facts.get("constraints", []) or []:
            rid = r.get("id")
            lines.append(
                f"- R{rid}: slack={r.get('slack')} | y={r.get('shadow_price')} | {r.get('expr')}"
            )
        return "REFERENCIAS_RESTRICCIONES:\n" + "\n".join(lines)
    return ""


def _facts_for_ai(module_key: str, facts: Dict[str, Any]) -> Dict[str, Any]:
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
    return facts


def _build_decision_prompt(module_key: str, facts: Dict[str, Any], context_text: str = "") -> str:
    facts_ai = _facts_for_ai(module_key, facts)
    facts_json = json.dumps(facts_ai, ensure_ascii=False, indent=2)
    refs = _human_refs(module_key, facts)

    module_rules = ""
    if module_key == "pl":
        module_rules = (
            "REGLAS (PL):\n"
            "- Conecta siempre un HECHO (y / slack / rc) con una decisión.\n"
            "- Interpreta holguras como recursos sobrantes o demanda insatisfecha según contexto.\n"
        )
    elif module_key == "transporte":
        module_rules = "REGLAS (Transporte): Sugiere usar celdas con costo reducido negativo si existen.\n"
    elif module_key == "redes":
        module_rules = "REGLAS (Redes): Menciona cuellos de botella (aristas saturadas).\n"

    context_instruction = ""
    if context_text and context_text.strip():
        context_instruction = (
            f"CONTEXTO DEL NEGOCIO / PROBLEMA:\n{context_text.strip()}\n\n"
            "INSTRUCCIÓN DE NARRATIVA:\n"
            "- Traduce los códigos (x1, R1) usando el contexto.\n"
            "- Ejemplo: Si x1=20 y contexto dice 'x1 son Sillas', escribe 'Se deben fabricar 20 Sillas'.\n"
            "- Ejemplo: Si R1 tiene slack=0, di que el recurso se agotó.\n\n"
        )

    return (
        "Eres un consultor experto en Investigación Operativa presentándole resultados a un Gerente.\n"
        "Tu trabajo es EXPLICAR la solución en lenguaje de negocios.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1) NO inventes números. Usa solo los de HECHOS_JSON.\n"
        "2) Usa el CONTEXTO DEL NEGOCIO para dar nombres reales.\n"
        "3) NO hagas preguntas.\n\n"
        + context_instruction
        + module_rules
        + "\nFORMATO DE RESPUESTA (Sección F obligatoria):\n"
        "F) Impacto en la toma de decisiones (IA)\n"
        "**Resumen Ejecutivo:**\n"
        "[Escribe aquí un párrafo narrativo y fluido. Ejemplo: 'Para maximizar la utilidad, el plan óptimo es producir 20 Sensores...']\n\n"
        "**Análisis de Recursos y Restricciones:**\n"
        "- [Viñeta interpretando holguras/slacks usando el contexto]\n"
        "- [Viñeta interpretando precios sombra/duales]\n\n"
        + refs
        + "\n\nHECHOS_JSON:\n"
        + facts_json
        + "\n"
    )


def deterministic_decision_addendum(module: str, facts: Dict[str, Any]) -> str:
    """Fallback offline (lo que sale cuando la IA falla)."""
    m = _module_key(module, facts)
    lines: List[str] = []
    lines.append("F) Impacto en la toma de decisiones (MODO OFFLINE - IA NO DISPONIBLE)")
    
    if m == "pl":
        cons = facts.get("constraints", []) or []
        active = [r for r in cons if abs(float(r.get("slack", 0))) < 1e-9]
        if active:
            top = active[0]
            lines.append(
                f"- Priorización: La restricción R{top.get('id')} está activa (se agotó el recurso o se cumplió el límite). "
                f"Tiene un precio sombra de {top.get('shadow_price')}, indicando su alto impacto marginal."
            )
        lines.append(f"- Verificación: is_optimal={facts.get('is_optimal')}")

    return "\n".join(lines).strip()


def _validate_decision_commentary(
    module: str, 
    facts: Dict[str, Any], 
    text: str, 
    context_text: str = ""
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not isinstance(text, str) or not text.strip():
        return False, ["Respuesta vacía."]

    low = _norm(text)
    if "f)" not in low:
        reasons.append("Falta encabezado F).")

    # Validar números (pasando contexto)
    ok_nums, nums_reasons = _validate_no_new_numbers(facts, text, context_text)
    if not ok_nums:
        reasons.extend(nums_reasons)

    return (len(reasons) == 0), reasons


def generate_decision_commentary(
    module: str,
    facts: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    max_retries: int = 1,
    context: str = "",
) -> str:
    """Genera F) con IA o fallback."""
    
    # 1. Verificar si tenemos cliente y clave
    if generate_text is None:
        print("DEBUG: Cliente gemini_client no importado.")
        return deterministic_decision_addendum(module, facts)

    key = _get_api_key(api_key)
    if not key:
        print("DEBUG: No se encontró API KEY en config.json ni variables de entorno.")
        return deterministic_decision_addendum(module, facts)

    mk = _module_key(module, facts)
    prompt = _build_decision_prompt(mk, facts, context_text=context)

    tries = max(0, int(max_retries))
    for attempt in range(tries + 1):
        try:
            print(f"DEBUG: Enviando solicitud a Gemini (Intento {attempt+1})...")
            text_out = generate_text(prompt, api_key=key)
            print("DEBUG: Respuesta recibida de Gemini.")
        except Exception as e:
            print(f"DEBUG: Error conectando con Gemini: {e}")
            text_out = ""

        ok, reasons = _validate_decision_commentary(module, facts, text_out, context_text=context)
        if ok:
            return text_out.strip()
        
        print(f"DEBUG: Respuesta rechazada por validación: {reasons}")
        
        # Reintento con prompt reforzado
        prompt += "\n\nIMPORTANTE: Tu respuesta anterior fue rechazada. NO inventes números. Devuelve solo F)."

    print("DEBUG: Se agotaron los reintentos. Usando modo offline.")
    return deterministic_decision_addendum(module, facts)


def generate_sensitivity_report(
    module: str,
    facts: Dict[str, Any],
    api_key: Optional[str] = None,
    max_retries: int = 1,
    context: str = "",
) -> str:
    base = deterministic_report(module, facts)
    f_text = generate_decision_commentary(
        module, 
        facts, 
        api_key=api_key, 
        max_retries=max_retries, 
        context=context
    )
    if f_text:
        return base + "\n\n" + f_text.strip()
    return base
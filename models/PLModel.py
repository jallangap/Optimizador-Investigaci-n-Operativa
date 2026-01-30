"""Programación Lineal (Python puro).

Este módulo reemplaza el uso de PuLP por implementaciones manuales:
  - Simplex (tableau)
  - Gran M
  - Dos Fases
  - Dualidad (construcción y resolución del dual)

La interfaz gráfica (views/PLView.py) envía expresiones lineales como strings,
por ejemplo:
  funcion_obj: "40*x1+30*x2"
  restriccion: "2*x1+x2<=100"

Notas:
  - Se asume x_j >= 0 (como en el proyecto original).
  - Se trabaja con floats y tolerancias para estabilidad.
  - NO se usa ninguna librería externa para resolver el modelo.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from utils.gemini_client import generate_text
from utils.sensitivity_ai import generate_sensitivity_report
from utils.sensitivity_context import set_last_facts


_TOL = 1e-9


class PLModel:
    def __init__(self):
        # API Key de Gemini (opcional):
        # - prioriza la variable de entorno GEMINI_API_KEY
        # - o `config.json` en la raíz del proyecto.
        # IMPORTANTE: **no** hardcodear llaves dentro del repositorio.
        try:
            from utils.gemini_client import get_api_key

            self.api_key = get_api_key()
        except Exception:
            self.api_key = None

    # ---------------------------------------------------------------------
    # Compatibilidad con PLController.py (aunque PLView llama directo al model)
    # ---------------------------------------------------------------------
    def maximizar(self, datos):
        return self.resolver_problema(datos, objetivo="Maximizar")

    def minimizar(self, datos):
        return self.resolver_problema(datos, objetivo="Minimizar")

    # ---------------------------------------------------------------------
    # Métodos expuestos a la UI
    # ---------------------------------------------------------------------
    def resolver_problema(self, datos, objetivo: str = "Maximizar"):
        """Resuelve un PL.

        En el proyecto original esta opción intentaba resolver incluso con >= o =
        usando el solver externo. En la versión manual:

        - Si el problema es estándar (todas <=), usa Simplex directo.
        - Si aparecen >= o =, se resuelve automáticamente con Dos Fases
          (para mantener funcionalidad sin obligar al usuario a cambiar UI).
        """
        try:
            p = _parse_problem(datos, objetivo)
            if all(s == "<=" for s in p.senses):
                return _solve_simplex_standard(p)
            # Soporte automático: si no es estándar, usamos Dos Fases.
            return _solve_two_phase(p)
        except Exception as e:
            return f"Error al resolver el problema: {str(e)}"

    def gran_m(self, datos, objetivo: str = "Maximizar"):
        try:
            p = _parse_problem(datos, objetivo)
            return _solve_big_m(p)
        except Exception as e:
            return f"Error al resolver el problema: {str(e)}"

    def dos_fases(self, datos, objetivo: str = "Maximizar"):
        try:
            p = _parse_problem(datos, objetivo)
            return _solve_two_phase(p)
        except Exception as e:
            return f"Error al resolver el problema en Dos Fases: {str(e)}"

    def dualidad(self, datos, objetivo: str = "Maximizar"):
        """Construye el dual (con reglas generales) y lo resuelve manualmente.

        Devuelve:
          - valores de x (primal)
          - Duals: valores de y (en el signo original del dual)
          - Valor Óptimo: valor óptimo del primal
        """
        try:
            primal = _parse_problem(datos, objetivo)
            primal_sol = _solve_two_phase(primal)
            if isinstance(primal_sol, str):
                return primal_sol

            dual = _build_dual_problem(primal)
            dual_sol = _solve_two_phase(dual)
            if isinstance(dual_sol, str):
                # Si por alguna razón falla el dual, igual devolvemos el primal.
                primal_sol["Duals"] = {"error": dual_sol}
                return primal_sol

            # Extraer y's del resultado dual (pueden venir como y1p/y1n, etc.)
            duals = _reconstruct_dual_values(dual, dual_sol)
            primal_sol["Duals"] = duals
            return primal_sol

        except Exception as e:
            return f"Error al resolver el problema en Dualidad: {str(e)}"

    def analizar_sensibilidad(self, resultado, datos):
        """Genera un análisis de sensibilidad asistido por IA, usando SOLO datos reales.

        Importante:
        - Holguras (slacks), precios sombra (y) y costos reducidos (rc) se calculan manualmente.
        - La IA únicamente interpreta esos valores, con guardrails para evitar alucinaciones.
        """
        try:
            funcion_obj = (datos.get("funcion_obj") or "").strip()
            restricciones = datos.get("restricciones") or []
            restricciones = [str(r).strip() for r in restricciones if str(r).strip()]
            objetivo = (datos.get("objetivo") or "Maximizar").strip()

            num_variables = datos.get("num_variables")
            try:
                num_variables = int(num_variables) if num_variables is not None else None
            except Exception:
                num_variables = None
            if not num_variables or num_variables <= 0:
                num_variables = _infer_num_vars(funcion_obj, restricciones)

            datos_lp = {
                "num_variables": num_variables,
                "funcion_obj": funcion_obj.replace(" ", ""),
                "restricciones": [r.replace(" ", "") for r in restricciones],
            }

            # Recalcular solución final (tableau) para extraer métricas duales reales.
            p = _parse_problem(datos_lp, objetivo)
            solve_out = _solve_for_sensitivity(p)
            if isinstance(solve_out, str):
                return solve_out
            sol, T = solve_out

            # Variables de decisión
            x_vals = {f"x{i+1}": float(sol.get(f"x{i+1}", 0.0)) for i in range(p.num_vars)}

            # Holguras / excesos reales respecto a las restricciones ingresadas por el usuario
            slacks = _compute_constraint_slacks(restricciones, x_vals, num_variables)

            # Precios sombra (dual prices) y costos reducidos desde la base final
            shadow_prices = _compute_shadow_prices(p, T)
            reduced_costs = _compute_reduced_costs(p, T, shadow_prices)

            # Certificado de optimalidad (según convención de costos reducidos)
            basis_names = {T.var_names[j] for j in T.basis}
            nonbasic = [name for name in T.var_names if name in reduced_costs and name not in basis_names]
            is_min = objetivo.strip().lower() == "minimizar"
            ok = True
            tol = 1e-9
            for name in nonbasic:
                rc = float(reduced_costs.get(name, 0.0))
                if is_min:
                    if rc < -tol:
                        ok = False
                        break
                else:
                    if rc > tol:
                        ok = False
                        break

            # Armar hechos (JSON) para Gemini + validación
            constraints_facts = []
            for i, info in enumerate(slacks, start=1):
                y = shadow_prices[i - 1] if i - 1 < len(shadow_prices) else None
                constraints_facts.append(
                    {
                        "id": i,
                        "expr": info.get("constraint"),
                        "type": info.get("type"),
                        "slack": float(info.get("value", 0.0)),
                        "shadow_price": (None if y is None else float(y)),
                    }
                )

            facts = {
                "module": "pl",
                "objective_sense": objetivo,
                "objective_expr": funcion_obj,
                "objective_value": float(sol.get("Valor Óptimo", 0.0)),
                "x": x_vals,
                "constraints": constraints_facts,
                "reduced_costs": {k: float(v) for k, v in reduced_costs.items()},
                "is_optimal": bool(ok),
            }

            # Guardar para la pestaña global de "Sensibilidad"
            set_last_facts("pl", facts)

            return generate_sensitivity_report("pl", facts, api_key=self.api_key, max_retries=1)
        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"


# ======================================================================
# Parser
# ======================================================================


@dataclass
class ParsedLP:
    num_vars: int
    # Maximizamos internamente siempre; si el usuario pidió minimizar, guardamos flag.
    is_min: bool
    c: List[float]                   # tamaño n
    A: List[List[float]]             # m x n
    b: List[float]                   # m
    senses: List[str]                # m, cada uno: "<=", ">=", "="
    # Para mostrar resultados similares al proyecto original
    # (slacks/excess/artificial por restricción)
    constraint_tags: List[Dict[str, str]]  # m: {"slack": "s1"} / {"excess":"e1"} / {"art":"a1"}


def _parse_problem(datos: Dict, objetivo: str) -> ParsedLP:
    n = int(datos["num_variables"])
    if n <= 0:
        raise ValueError("El número de variables debe ser mayor que 0")

    func = _normalize_expr(datos["funcion_obj"])
    c, const = _parse_linear_expr(func, n)
    if abs(const) > _TOL:
        # No es error fatal; simplemente lo ignoramos en el motor (se suma al z al final).
        # Se mantiene este comportamiento por compatibilidad.
        pass

    is_min = (objetivo.strip().lower() == "minimizar")
    # Internamente resolvemos como max. Si es min, multiplicamos por -1.
    if is_min:
        c = [-v for v in c]

    A: List[List[float]] = []
    b: List[float] = []
    senses: List[str] = []
    tags: List[Dict[str, str]] = []

    restricciones = datos.get("restricciones", [])
    if not restricciones:
        raise ValueError("Debe ingresar al menos una restricción")

    for i, r in enumerate(restricciones, start=1):
        r = _normalize_constraint(r)
        lhs, sense, rhs = _split_constraint(r)
        a_row, lhs_const = _parse_linear_expr(lhs, n)
        if abs(lhs_const) > _TOL:
            # Llevamos constantes al RHS.
            rhs_val = _parse_number(rhs) - lhs_const
        else:
            rhs_val = _parse_number(rhs)

        # Normalizar RHS >= 0 (multiplicar por -1 si hace falta)
        if rhs_val < -_TOL:
            rhs_val = -rhs_val
            a_row = [-v for v in a_row]
            if sense == "<=":
                sense = ">="
            elif sense == ">=":
                sense = "<="
            # "=" se mantiene

        A.append(a_row)
        b.append(rhs_val)
        senses.append(sense)

        t: Dict[str, str] = {}
        if sense == "<=":
            t["slack"] = f"s{i}"
        elif sense == ">=":
            t["excess"] = f"e{i}"
            t["art"] = f"a{i}"
        elif sense == "=":
            t["art"] = f"a{i}"
        tags.append(t)

    return ParsedLP(
        num_vars=n,
        is_min=is_min,
        c=c,
        A=A,
        b=b,
        senses=senses,
        constraint_tags=tags,
    )


def _normalize_expr(expr: str) -> str:
    return (expr or "").strip().replace(" ", "")


def _normalize_constraint(c: str) -> str:
    c = (c or "").strip().replace(" ", "")
    return c.replace("≤", "<=").replace("≥", ">=")


def _split_constraint(text: str) -> Tuple[str, str, str]:
    for op in ("<=", ">=", "="):
        if op in text:
            lhs, rhs = text.split(op, 1)
            return lhs, op, rhs
    raise ValueError(f"Formato de restricción inválido: {text}")


_TERM_RE = re.compile(r"^([+-]?\d*(?:\.\d+)?(?:e[+-]?\d+)?)?\*?x(\d+)$", re.IGNORECASE)
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def _parse_number(s: str) -> float:
    s = (s or "").strip()
    if not _NUM_RE.match(s):
        raise ValueError(f"RHS debe ser un número. Recibido: {s}")
    return float(s)


def _parse_linear_expr(expr: str, num_vars: int) -> Tuple[List[float], float]:
    """Convierte una expresión lineal en coeficientes.

    Soporta términos como: 2*x1, -x2, +3.5*x3, x4.
    También permite constantes (ej: +10).
    """
    expr = _normalize_expr(expr)
    if not expr:
        raise ValueError("Expresión vacía")

    # Para dividir por '+', primero normalizamos signos.
    # Ej: "2*x1-x2+3" => "2*x1+-x2+3"
    expr_norm = expr.replace("-", "+-")
    if expr_norm.startswith("+-"):
        expr_norm = "-" + expr_norm[2:]

    coeffs = [0.0] * num_vars
    constant = 0.0
    for term in expr_norm.split("+"):
        if term == "" or term == "+":
            continue

        m = _TERM_RE.match(term)
        if m:
            coef_s, idx_s = m.group(1), m.group(2)
            idx = int(idx_s)
            if idx < 1 or idx > num_vars:
                raise ValueError(f"Variable fuera de rango: x{idx}")

            if coef_s is None or coef_s in ("", "+"):
                coef = 1.0
            elif coef_s == "-":
                coef = -1.0
            else:
                coef = float(coef_s)
            coeffs[idx - 1] += coef
            continue

        # Constante
        if _NUM_RE.match(term):
            constant += float(term)
            continue

        raise ValueError(
            "Solo se admiten expresiones lineales del tipo a*x1+b*x2... sin paréntesis. "
            f"Término inválido: {term} (expr: {expr})"
        )

    return coeffs, constant


# ======================================================================
# Motor Simplex (tableau)
# ======================================================================


@dataclass
class Tableau:
    # Matriz (m+1) x (n+1). Última columna: RHS.
    tab: List[List[float]]
    basis: List[int]               # índices de variables básicas (tamaño m)
    var_names: List[str]           # nombres columnas (tamaño n)

    def m(self) -> int:
        return len(self.basis)

    def n(self) -> int:
        return len(self.var_names)


def _pivot(T: Tableau, row: int, col: int) -> None:
    tab = T.tab
    m = T.m()
    n = T.n()
    pivot_val = tab[row][col]
    if abs(pivot_val) < _TOL:
        raise ZeroDivisionError("Pivote casi cero")

    # Normalizar fila pivote
    inv = 1.0 / pivot_val
    for j in range(n + 1):
        tab[row][j] *= inv

    # Eliminar columna pivote en otras filas
    for i in range(m + 1):
        if i == row:
            continue
        factor = tab[i][col]
        if abs(factor) < _TOL:
            continue
        for j in range(n + 1):
            tab[i][j] -= factor * tab[row][j]

    T.basis[row] = col


def _choose_entering(T: Tableau) -> Optional[int]:
    """Regla de Bland: menor índice con coeficiente negativo en la fila objetivo."""
    obj = T.tab[T.m()]
    for j in range(T.n()):
        if obj[j] < -_TOL:
            return j
    return None


def _choose_leaving(T: Tableau, entering: int) -> Optional[int]:
    m = T.m()
    tab = T.tab
    best_row = None
    best_ratio = None
    for i in range(m):
        a = tab[i][entering]
        if a > _TOL:
            ratio = tab[i][-1] / a
            if ratio < -_TOL:
                continue
            if best_ratio is None or ratio < best_ratio - _TOL or (
                abs(ratio - best_ratio) <= _TOL and i < (best_row or 10**9)
            ):
                best_ratio = ratio
                best_row = i
    return best_row


def _simplex(T: Tableau, max_iters: int = 10_000) -> Tuple[str, Tableau]:
    """Ejecuta simplex (maximización)."""
    for _ in range(max_iters):
        entering = _choose_entering(T)
        if entering is None:
            return "Óptimo", T

        leaving = _choose_leaving(T, entering)
        if leaving is None:
            return "No acotado", T

        _pivot(T, leaving, entering)

    return "Iteraciones máximas alcanzadas", T


def _canonicalize_objective(T: Tableau) -> None:
    """Ajusta la fila objetivo para que coeficientes de variables básicas queden en 0."""
    m = T.m()
    obj = T.tab[m]
    for i in range(m):
        bv = T.basis[i]
        coef = obj[bv]
        if abs(coef) < _TOL:
            continue
        # obj = obj - coef * row_i (pero como row_i tiene bv=1, elimina coef)
        row = T.tab[i]
        for j in range(T.n() + 1):
            obj[j] -= coef * row[j]


def _extract_solution(T: Tableau) -> Dict[str, float]:
    m = T.m()
    n = T.n()
    sol = {name: 0.0 for name in T.var_names}
    for i in range(m):
        sol[T.var_names[T.basis[i]]] = T.tab[i][-1]
    # limpiar -0.0
    for k, v in list(sol.items()):
        if abs(v) < 1e-12:
            sol[k] = 0.0
    return sol


# ======================================================================
# Construcción de tableau (estándar, Gran M, Dos Fases)
# ======================================================================


def _build_initial_tableau(p: ParsedLP, method: str, M: float = 1e6) -> Tuple[Tableau, Dict[str, int], Dict[str, str]]:
    """Crea tableau inicial.

    method:
      - "standard": solo <= (slacks)
      - "bigm": incluye artificiales y penalización M
      - "phase1": objetivo -sum(a)

    Retorna:
      - Tableau
      - index map var_name -> column index
      - meta map: var_name -> kind ("x"|"s"|"e"|"a")
    """
    m = len(p.A)
    n = p.num_vars

    var_names: List[str] = [f"x{i+1}" for i in range(n)]
    var_kind: Dict[str, str] = {f"x{i+1}": "x" for i in range(n)}

    # Crear nombres de variables adicionales por restricción
    slack_names: List[Optional[str]] = [None] * m
    excess_names: List[Optional[str]] = [None] * m
    art_names: List[Optional[str]] = [None] * m

    for i in range(m):
        t = p.constraint_tags[i]
        if "slack" in t:
            slack_names[i] = t["slack"]
        if "excess" in t:
            excess_names[i] = t["excess"]
        if "art" in t:
            art_names[i] = t["art"]

    # Orden de columnas: x, s, e, a
    for nm in slack_names:
        if nm is not None:
            var_names.append(nm)
            var_kind[nm] = "s"
    for nm in excess_names:
        if nm is not None:
            var_names.append(nm)
            var_kind[nm] = "e"
    for nm in art_names:
        if nm is not None:
            var_names.append(nm)
            var_kind[nm] = "a"

    idx = {name: j for j, name in enumerate(var_names)}
    total_vars = len(var_names)

    # Tabla: m restricciones + 1 objetivo
    tab = [[0.0] * (total_vars + 1) for _ in range(m + 1)]
    basis: List[int] = [0] * m

    # Llenar restricciones
    for i in range(m):
        # x
        for j in range(n):
            tab[i][j] = float(p.A[i][j])
        # extras
        t = p.constraint_tags[i]
        if p.senses[i] == "<=":
            s = t.get("slack")
            if s is None:
                raise ValueError("Falta slack tag")
            tab[i][idx[s]] = 1.0
            basis[i] = idx[s]
        elif p.senses[i] == ">=":
            e = t.get("excess")
            a = t.get("art")
            if e is None or a is None:
                raise ValueError("Faltan tags para >=")
            tab[i][idx[e]] = -1.0
            tab[i][idx[a]] = 1.0
            basis[i] = idx[a]
        elif p.senses[i] == "=":
            a = t.get("art")
            if a is None:
                raise ValueError("Falta art tag para =")
            tab[i][idx[a]] = 1.0
            basis[i] = idx[a]
        else:
            raise ValueError(f"Sentido inválido: {p.senses[i]}")

        tab[i][-1] = float(p.b[i])

    # Objetivo
    obj = tab[m]
    if method == "standard" or method == "bigm":
        # Max: z - c^T x = 0  => coef = -c
        for j in range(n):
            obj[j] = -float(p.c[j])

        # Penalización Gran M para artificiales
        if method == "bigm":
            for name in var_names:
                if var_kind[name] == "a":
                    # En maximización: c_a = -M  => coef = -c_a = +M
                    obj[idx[name]] = +M

    elif method == "phase1":
        # Fase 1: maximizar -sum(a)
        for name in var_names:
            if var_kind[name] == "a":
                # c_a = -1  => coef = -c_a = +1
                obj[idx[name]] = +1.0
    else:
        raise ValueError(f"Método de tableau desconocido: {method}")

    # Canonicalizar con la base inicial
    T = Tableau(tab=tab, basis=basis, var_names=var_names)
    _canonicalize_objective(T)

    return T, idx, var_kind


def _remove_columns(T: Tableau, cols_to_remove: List[int]) -> Tableau:
    cols_to_remove = sorted(set(cols_to_remove))
    keep = [j for j in range(T.n()) if j not in cols_to_remove]
    new_names = [T.var_names[j] for j in keep]

    # Mapping old->new
    mapping = {old: new for new, old in enumerate(keep)}
    new_tab = []
    for row in T.tab:
        new_row = [row[j] for j in keep] + [row[-1]]
        new_tab.append(new_row)
    new_basis = [mapping[b] for b in T.basis]
    return Tableau(tab=new_tab, basis=new_basis, var_names=new_names)


def _drop_rows(T: Tableau, rows_to_drop: List[int]) -> Tableau:
    rows_to_drop = sorted(set(rows_to_drop))
    if not rows_to_drop:
        return T
    m = T.m()
    keep_rows = [i for i in range(m) if i not in rows_to_drop]
    new_tab = [T.tab[i] for i in keep_rows] + [T.tab[m]]
    new_basis = [T.basis[i] for i in keep_rows]
    return Tableau(tab=new_tab, basis=new_basis, var_names=T.var_names)


# ======================================================================
# Solvers
# ======================================================================


def _solve_simplex_standard(p: ParsedLP):
    # Solo <=, no artificial
    T, _, _ = _build_initial_tableau(p, method="standard")
    status, T = _simplex(T)
    if status != "Óptimo":
        return f"Problema {status}"
    sol = _extract_solution(T)
    return _format_result(p, sol, objective_value=T.tab[T.m()][-1])


def _solve_big_m(p: ParsedLP, M: float = 1e6):
    T, _, _ = _build_initial_tableau(p, method="bigm", M=M)
    status, T = _simplex(T)
    if status != "Óptimo":
        return f"Problema {status}"
    sol = _extract_solution(T)
    # Verificar que artificiales quedaron ~0
    for k, v in sol.items():
        if k.startswith("a") and v > 1e-6:
            return "Problema infactible (artificiales > 0)"
    return _format_result(p, sol, objective_value=T.tab[T.m()][-1])


def _solve_two_phase(p: ParsedLP):
    # Fase 1
    T1, _, kind = _build_initial_tableau(p, method="phase1")
    status1, T1 = _simplex(T1)
    if status1 != "Óptimo":
        return f"Fase 1: {status1}"
    phase1_obj = T1.tab[T1.m()][-1]  # max -sum(a)
    if abs(phase1_obj) > 1e-6:
        return "Problema infactible: suma de artificiales > 0 en Fase 1"

    # Intentar sacar artificiales de la base
    art_cols = [j for j, name in enumerate(T1.var_names) if kind.get(name) == "a"]
    rows_to_drop: List[int] = []
    for i in range(T1.m()):
        bv = T1.basis[i]
        if bv in art_cols:
            # Buscar una columna no artificial para pivotear
            pivot_col = None
            for j in range(T1.n()):
                if j in art_cols:
                    continue
                if abs(T1.tab[i][j]) > _TOL:
                    pivot_col = j
                    break
            if pivot_col is not None:
                _pivot(T1, i, pivot_col)
            else:
                # Fila redundante: no hay con qué pivotear
                rows_to_drop.append(i)

    T1 = _drop_rows(T1, rows_to_drop)

    # Remover columnas artificiales
    if art_cols:
        T2 = _remove_columns(T1, art_cols)
    else:
        T2 = T1

    # Fase 2: objetivo original
    # Reconstruimos costos para variables que quedaron
    n_original = p.num_vars
    costs: Dict[str, float] = {f"x{i+1}": float(p.c[i]) for i in range(n_original)}
    # slacks/excess tienen costo 0
    obj = T2.tab[T2.m()]
    for j, name in enumerate(T2.var_names):
        obj[j] = -costs.get(name, 0.0)
    obj[-1] = 0.0
    _canonicalize_objective(T2)

    status2, T2 = _simplex(T2)
    if status2 != "Óptimo":
        return f"Fase 2: {status2}"

    sol = _extract_solution(T2)
    return _format_result(p, sol, objective_value=T2.tab[T2.m()][-1])


def _format_result(p: ParsedLP, sol: Dict[str, float], objective_value: float) -> Dict[str, float]:
    # Completar variables esperadas (x, s/e/a por restricción)
    resultado: Dict[str, float] = {}
    for i in range(p.num_vars):
        name = f"x{i+1}"
        resultado[name] = float(sol.get(name, 0.0))

    # Extras por restricción según tags
    for t in p.constraint_tags:
        for k in ("slack", "excess", "art"):
            if k in t:
                nm = t[k]
                resultado[nm] = float(sol.get(nm, 0.0))

    # Valor óptimo del problema ORIGINAL
    # Recordatorio: si el usuario pidió minimizar, internamente maximizamos -c.
    z = objective_value
    if p.is_min:
        z = -z
    # limpiar -0.0
    if abs(z) < 1e-12:
        z = 0.0
    resultado["Valor Óptimo"] = float(z)
    return resultado


# ======================================================================
# Dualidad
# ======================================================================


def _build_dual_problem(primal: ParsedLP) -> ParsedLP:
    """Construye el dual usando reglas generales.

    Convenciones:
      - Primal: x >= 0
      - Restricciones: <=, >=, =
      - Objetivo: max o min
    """
    m = len(primal.A)
    n = primal.num_vars

    # El primal ya está en forma de maximización interna. Para reglas del dual,
    # necesitamos saber si el usuario quería max o min.
    primal_is_min = primal.is_min

    # Dual variable signs según primal constraints
    # Si primal original era MAX:
    #   <= => y>=0, >= => y<=0, = => y libre
    # Si primal original era MIN:
    #   >= => y>=0, <= => y<=0, = => y libre
    # Pero como internamente siempre maximizamos, invertimos con primal_is_min.
    # Si primal_is_min=True, significa que el usuario quería MIN, pero internamente
    # ya lo transformamos a MAX. Para el dual, usamos reglas del problema ORIGINAL.

    # Estructura de variables y:
    #   y>=0 => yk
    #   y<=0 => yk = -ykp, ykp>=0
    #   y libre => yk = ykp - ykn
    y_names: List[str] = []
    y_decomp: List[Tuple[str, str, str]] = []
    # cada entrada: (type, pos_name, neg_name)
    # type: 'pos'|'neg'|'free'

    for i, s in enumerate(primal.senses, start=1):
        if not primal_is_min:  # primal original MAX
            if s == "<=":
                y_decomp.append(("pos", f"y{i}", ""))
                y_names.append(f"y{i}")
            elif s == ">=":
                # y <= 0 => y = -ypos
                y_decomp.append(("neg", f"y{i}p", ""))
                y_names.append(f"y{i}p")
            else:  # "=" free
                y_decomp.append(("free", f"y{i}p", f"y{i}n"))
                y_names.extend([f"y{i}p", f"y{i}n"])
        else:  # primal original MIN
            if s == ">=":
                y_decomp.append(("pos", f"y{i}", ""))
                y_names.append(f"y{i}")
            elif s == "<=":
                y_decomp.append(("neg", f"y{i}p", ""))
                y_names.append(f"y{i}p")
            else:
                y_decomp.append(("free", f"y{i}p", f"y{i}n"))
                y_names.extend([f"y{i}p", f"y{i}n"])

    # Dual objective:
    # - Si primal MAX => dual MIN: min b^T y
    # - Si primal MIN => dual MAX: max b^T y
    dual_is_min = (not primal_is_min)

    # Construimos función objetivo en forma de string (solo para consistencia en reportes)
    # y guardamos c numérico.
    dual_c: List[float] = []
    # Reglas al descomponer variables:
    # y<=0: y = -ypos => b_i*y = -b_i*ypos
    # y libre: y = y+ - y- => b_i*y = b_i*y+ - b_i*y-
    for i in range(m):
        typ, pos, neg = y_decomp[i]
        bi = primal.b[i]
        if typ == "pos":
            dual_c.append(bi)
        elif typ == "neg":
            dual_c.append(-bi)
        else:  # free
            dual_c.append(bi)
            dual_c.append(-bi)

    # Dual constraints:
    # - Si primal MAX => A^T y >= c
    # - Si primal MIN => A^T y <= c
    # OJO: primal.c aquí corresponde a la forma interna (si primal original era MIN,
    #      primal.c ya está negada). Para el dual del ORIGINAL, necesitamos c_original.
    c_original = primal.c if not primal_is_min else [-v for v in primal.c]

    dual_sense = ">=" if not primal_is_min else "<="

    # Construir matriz A_dual (n restricciones, |y_names| variables)
    A_dual: List[List[float]] = []
    b_dual: List[float] = []
    senses_dual: List[str] = []
    tags_dual: List[Dict[str, str]] = []

    # Para cada variable x_j del primal, construir restricción dual
    # sum_i a_ij y_i (con descomposición) (>= o <=) c_j
    for j in range(n):
        row: List[float] = []
        for i in range(m):
            aij = primal.A[i][j]
            typ, _, _ = y_decomp[i]
            if typ == "pos":
                row.append(aij)
            elif typ == "neg":
                # y = -ypos
                row.append(-aij)
            else:
                # y = y+ - y-
                row.append(aij)
                row.append(-aij)
        A_dual.append(row)
        b_dual.append(c_original[j])
        senses_dual.append(dual_sense)
        # Tags para resultados estilo proyecto (todas serán <= o >= según dual_sense)
        t: Dict[str, str] = {}
        if dual_sense == "<=":
            t["slack"] = f"s{j+1}"
        else:
            t["excess"] = f"e{j+1}"
            t["art"] = f"a{j+1}"
        tags_dual.append(t)

    # Si el dual es MIN, internamente convertiremos a MAX negando dual_c.
    dual_objetivo = "Minimizar" if dual_is_min else "Maximizar"
    dual_datos = {
        "num_variables": len(dual_c),
        "funcion_obj": "0",  # no se usa en este punto
        "restricciones": [],
    }

    # Construir ParsedLP directamente (sin pasar por parser de strings)
    # Internamente guardamos dual_is_min en is_min.
    dual_parsed = ParsedLP(
        num_vars=len(dual_c),
        is_min=dual_is_min,
        c=(dual_c if not dual_is_min else [-v for v in dual_c]),  # interna siempre max
        A=A_dual,
        b=b_dual,
        senses=senses_dual,
        constraint_tags=tags_dual,
    )
    return dual_parsed


def _reconstruct_dual_values(dual: ParsedLP, dual_sol: Dict[str, float]) -> Dict[str, float]:
    """Reconstruye y_i originales a partir de variables descompuestas."""
    # Las variables del dual están nombradas como x1..xn en el motor.
    # Pero en _build_dual_problem construimos dual_parsed con num_vars=len(dual_c)
    # y por tanto las variables se llaman x1..xK.
    # Para devolver algo legible, retornamos y1..ym si es posible.

    # Simple: devolvemos los x del dual como y_k.
    duals: Dict[str, float] = {}
    for i in range(dual.num_vars):
        duals[f"y{i+1}"] = float(dual_sol.get(f"x{i+1}", 0.0))
    return duals


# ======================================================================
# Sensibilidad (extracción numérica real desde el tableau final)
# ======================================================================


def _infer_num_vars(funcion_obj: str, restricciones: List[str]) -> int:
    """Infere num_variables buscando patrones x1, x2, ... en el texto."""
    max_idx = 0
    text = (funcion_obj or "") + " " + " ".join(restricciones or [])
    for m in re.finditer(r"x(\d+)", text):
        try:
            max_idx = max(max_idx, int(m.group(1)))
        except Exception:
            continue
    if max_idx <= 0:
        raise ValueError("No se pudo inferir el número de variables. Ingresa 'Número de variables'.")
    return max_idx


def _solve_for_sensitivity(p: ParsedLP):
    """Resuelve y retorna (solución, tableau_final) para extracción de sensibilidad."""
    if all(s == "<=" for s in p.senses):
        # Standard simplex tableau
        T, _, _ = _build_initial_tableau(p, method="standard")
        status, T = _simplex(T)
        if status != "Óptimo":
            return f"Problema {status}"
        sol = _extract_solution(T)
        out = _format_result(p, sol, objective_value=T.tab[T.m()][-1])
        return out, T
    # Caso general: dos fases (retornando tableau de fase 2)
    out, T2 = _solve_two_phase_with_tableau(p)
    if isinstance(out, str):
        return out
    return out, T2


def _solve_two_phase_with_tableau(p: ParsedLP):
    """Dos fases, pero devolviendo el tableau final de la Fase 2."""
    T1, _, kind = _build_initial_tableau(p, method="phase1")
    status1, T1 = _simplex(T1)
    if status1 != "Óptimo":
        return f"Fase 1: {status1}", None
    phase1_obj = T1.tab[T1.m()][-1]
    if abs(phase1_obj) > 1e-6:
        return "Problema infactible: suma de artificiales > 0 en Fase 1", None

    # Sacar artificiales de la base / eliminar filas redundantes
    art_cols = [j for j, name in enumerate(T1.var_names) if kind.get(name) == "a"]
    rows_to_drop: List[int] = []
    for i in range(T1.m()):
        bv = T1.basis[i]
        if bv in art_cols:
            pivot_col = None
            for j in range(T1.n()):
                if j in art_cols:
                    continue
                if abs(T1.tab[i][j]) > _TOL:
                    pivot_col = j
                    break
            if pivot_col is not None:
                _pivot(T1, i, pivot_col)
            else:
                rows_to_drop.append(i)

    T1 = _drop_rows(T1, rows_to_drop)
    if art_cols:
        T2 = _remove_columns(T1, art_cols)
    else:
        T2 = T1

    # Fase 2: objetivo original (interno ya es max)
    n_original = p.num_vars
    costs: Dict[str, float] = {f"x{i+1}": float(p.c[i]) for i in range(n_original)}
    obj = T2.tab[T2.m()]
    for j, name in enumerate(T2.var_names):
        obj[j] = -costs.get(name, 0.0)
    obj[-1] = 0.0
    _canonicalize_objective(T2)

    status2, T2 = _simplex(T2)
    if status2 != "Óptimo":
        return f"Fase 2: {status2}", None

    sol = _extract_solution(T2)
    out = _format_result(p, sol, objective_value=T2.tab[T2.m()][-1])
    return out, T2


def _compute_constraint_slacks(constraints: List[str], x_vals: Dict[str, float], num_vars: int) -> List[Dict[str, float]]:
    """Calcula holgura/exceso respecto a las restricciones originales del usuario."""
    out: List[Dict[str, float]] = []
    for raw in constraints:
        c = _normalize_constraint(raw)
        lhs, sense, rhs = _split_constraint(c)
        a, lhs_const = _parse_linear_expr(lhs, num_vars)
        rhs_val = _parse_number(rhs) - lhs_const
        lhs_val = 0.0
        for j in range(num_vars):
            lhs_val += a[j] * float(x_vals.get(f"x{j+1}", 0.0))

        if sense == "<=":
            val = rhs_val - lhs_val
            typ = "Slack"
        elif sense == ">=":
            val = lhs_val - rhs_val
            typ = "Surplus"
        else:
            val = rhs_val - lhs_val
            typ = "Equality"

        if abs(val) < 1e-10:
            val = 0.0
        out.append({"constraint": raw, "type": typ, "value": float(val)})
    return out


def _compute_shadow_prices(p: ParsedLP, T: Tableau) -> List[Optional[float]]:
    """Calcula precios sombra y resolviendo B^T y = c_B (con c del problema ORIGINAL)."""
    m = len(p.A)
    if T.m() != m:
        # Caso raro (filas redundantes eliminadas). No arriesgamos valores incorrectos.
        return [None] * m

    # c del problema original (si original era MIN, internamente negamos)
    c_original = p.c if not p.is_min else [-v for v in p.c]

    # Construir columnas de la matriz aumentada (x, slacks, excess) según nombres del tableau
    col_vectors: Dict[str, List[float]] = {}
    for name in T.var_names:
        vec = [0.0] * m
        if name.startswith("x"):
            j = int(name[1:]) - 1
            for i in range(m):
                vec[i] = float(p.A[i][j])
        elif name.startswith("s"):
            i = int(name[1:]) - 1
            if 0 <= i < m:
                vec[i] = 1.0
        elif name.startswith("e"):
            i = int(name[1:]) - 1
            if 0 <= i < m:
                vec[i] = -1.0
        else:
            # artificial ya no debería existir aquí
            pass
        col_vectors[name] = vec

    # Matriz base B (m x m) a partir de variables básicas
    basis_names = [T.var_names[j] for j in T.basis]
    B = [[0.0] * m for _ in range(m)]
    cB = [0.0] * m
    for k, bname in enumerate(basis_names):
        col = col_vectors.get(bname)
        if col is None:
            return [None] * m
        for i in range(m):
            B[i][k] = col[i]
        if bname.startswith("x"):
            idx = int(bname[1:]) - 1
            cB[k] = float(c_original[idx])
        else:
            cB[k] = 0.0

    # Resolver B^T y = cB  => (B^T) y = cB
    BT = [[B[j][i] for j in range(m)] for i in range(m)]
    y = _solve_linear_system(BT, cB)
    if y is None:
        return [None] * m
    return [float(v) for v in y]


def _compute_reduced_costs(p: ParsedLP, T: Tableau, shadow_prices: List[Optional[float]]) -> Dict[str, float]:
    """Costos reducidos: rc_j = c_j - y^T a_j (c del problema ORIGINAL)."""
    m = len(p.A)
    c_original = p.c if not p.is_min else [-v for v in p.c]
    if any(v is None for v in shadow_prices) or len(shadow_prices) != m:
        # Sin y no hay costos reducidos confiables
        out = {}
        for name in T.var_names:
            if name.startswith("x") or name.startswith("s") or name.startswith("e"):
                out[name] = 0.0
        return out

    y = [float(v) for v in shadow_prices]  # type: ignore

    def dot(u, v):
        s = 0.0
        for a, b in zip(u, v):
            s += a * b
        return s

    # Columnas a_j según nombres
    out: Dict[str, float] = {}
    for name in T.var_names:
        if name.startswith("x"):
            j = int(name[1:]) - 1
            a = [float(p.A[i][j]) for i in range(m)]
            cj = float(c_original[j])
        elif name.startswith("s"):
            i = int(name[1:]) - 1
            a = [0.0] * m
            if 0 <= i < m:
                a[i] = 1.0
            cj = 0.0
        elif name.startswith("e"):
            i = int(name[1:]) - 1
            a = [0.0] * m
            if 0 <= i < m:
                a[i] = -1.0
            cj = 0.0
        else:
            continue

        rc = cj - dot(y, a)
        if abs(rc) < 1e-10:
            rc = 0.0
        out[name] = float(rc)

    # Las variables básicas deben tener rc=0 por definición
    for bcol in T.basis:
        bname = T.var_names[bcol]
        if bname in out:
            out[bname] = 0.0
    return out


def _solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Resuelve A x = b con eliminación gaussiana (Python puro)."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        return None

    # Copias
    M = [row[:] + [float(b[i])] for i, row in enumerate(A)]

    for col in range(n):
        # Pivot parcial
        pivot = col
        best = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > best:
                best = abs(M[r][col])
                pivot = r
        if best < 1e-12:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        # Normalizar fila pivote
        pv = M[col][col]
        inv = 1.0 / pv
        for j in range(col, n + 1):
            M[col][j] *= inv

        # Eliminar
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]

    return [float(M[i][n]) for i in range(n)]

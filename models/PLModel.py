"""Programación Lineal (Python puro).

Este módulo reemplaza el uso de PuLP por implementaciones manuales:
  - Simplex (tableau)
  - Gran M
  - Dos Fases
  - Dualidad (construcción y resolución del dual)

La interfaz gráfica (views/PLView.py) envía expresiones lineales como strings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Importaciones de utilidades propias
from utils.gemini_client import get_api_key
from utils.sensitivity_ai import generate_sensitivity_report


_TOL = 1e-9


class PLModel:
    def __init__(self):
        # API Key de Gemini: prioriza la variable de entorno GEMINI_API_KEY.
        self.api_key = get_api_key()

    # ---------------------------------------------------------------------
    # Compatibilidad con PLController.py
    # ---------------------------------------------------------------------
    def maximizar(self, datos):
        return self.resolver_problema(datos, objetivo="Maximizar")

    def minimizar(self, datos):
        return self.resolver_problema(datos, objetivo="Minimizar")

    # ---------------------------------------------------------------------
    # Métodos expuestos a la UI
    # ---------------------------------------------------------------------
    def resolver_problema(self, datos, objetivo: str = "Maximizar"):
        """Resuelve un PL."""
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
        try:
            primal = _parse_problem(datos, objetivo)
            primal_sol = _solve_two_phase(primal)
            if isinstance(primal_sol, str):
                return primal_sol

            dual = _build_dual_problem(primal)
            dual_sol = _solve_two_phase(dual)
            if isinstance(dual_sol, str):
                primal_sol["Duals"] = {"error": dual_sol}
                return primal_sol

            duals = _reconstruct_dual_values(dual, dual_sol)
            primal_sol["Duals"] = duals
            return primal_sol

        except Exception as e:
            return f"Error al resolver el problema en Dualidad: {str(e)}"

    def analizar_sensibilidad(self, resultado, datos):
        """Genera un análisis de sensibilidad asistido por IA."""
        try:
            # 1. Recuperar datos básicos
            funcion_obj = (datos.get("funcion_obj") or "").strip()
            restricciones = datos.get("restricciones") or []
            restricciones = [str(r).strip() for r in restricciones if str(r).strip()]
            objetivo = (datos.get("objetivo") or "Maximizar").strip()
            
            # --- Recuperar el contexto del negocio ---
            contexto_usuario = datos.get("contexto", "")

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

            # 2. Recalcular solución final (tableau) para extraer métricas duales reales.
            p = _parse_problem(datos_lp, objetivo)
            solve_out = _solve_for_sensitivity(p)
            if isinstance(solve_out, str):
                return solve_out
            sol, T = solve_out

            # 3. Variables de decisión
            x_vals = {f"x{i+1}": float(sol.get(f"x{i+1}", 0.0)) for i in range(p.num_vars)}

            # 4. Holguras
            slacks = _compute_constraint_slacks(restricciones, x_vals, num_variables)

            # 5. Precios sombra y costos reducidos
            shadow_prices = _compute_shadow_prices(p, T)
            reduced_costs = _compute_reduced_costs(p, T, shadow_prices)

            # 6. Certificado de optimalidad
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

            # 7. Armar hechos (JSON)
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

            # 8. Llamada final con contexto
            return generate_sensitivity_report(
                'pl', 
                facts, 
                api_key=self.api_key, 
                max_retries=1, 
                context=contexto_usuario
            )

        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"


# ======================================================================
# Parser
# ======================================================================


@dataclass
class ParsedLP:
    num_vars: int
    is_min: bool
    c: List[float]
    A: List[List[float]]
    b: List[float]
    senses: List[str]
    constraint_tags: List[Dict[str, str]]


def _parse_problem(datos: Dict, objetivo: str) -> ParsedLP:
    n = int(datos["num_variables"])
    if n <= 0:
        raise ValueError("El número de variables debe ser mayor que 0")

    func = _normalize_expr(datos["funcion_obj"])
    c, const = _parse_linear_expr(func, n)
    
    is_min = (objetivo.strip().lower() == "minimizar")
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
            rhs_val = _parse_number(rhs) - lhs_const
        else:
            rhs_val = _parse_number(rhs)

        if rhs_val < -_TOL:
            rhs_val = -rhs_val
            a_row = [-v for v in a_row]
            if sense == "<=":
                sense = ">="
            elif sense == ">=":
                sense = "<="

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
    expr = _normalize_expr(expr)
    if not expr:
        raise ValueError("Expresión vacía")
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

        if _NUM_RE.match(term):
            constant += float(term)
            continue

        raise ValueError(f"Término inválido: {term}")

    return coeffs, constant


# ======================================================================
# Motor Simplex (tableau)
# ======================================================================


@dataclass
class Tableau:
    tab: List[List[float]]
    basis: List[int]
    var_names: List[str]

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

    inv = 1.0 / pivot_val
    for j in range(n + 1):
        tab[row][j] *= inv

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
    m = T.m()
    obj = T.tab[m]
    for i in range(m):
        bv = T.basis[i]
        coef = obj[bv]
        if abs(coef) < _TOL:
            continue
        row = T.tab[i]
        for j in range(T.n() + 1):
            obj[j] -= coef * row[j]


def _extract_solution(T: Tableau) -> Dict[str, float]:
    m = T.m()
    sol = {name: 0.0 for name in T.var_names}
    for i in range(m):
        sol[T.var_names[T.basis[i]]] = T.tab[i][-1]
    for k, v in list(sol.items()):
        if abs(v) < 1e-12:
            sol[k] = 0.0
    return sol


# ======================================================================
# Construcción de tableau
# ======================================================================


def _build_initial_tableau(p: ParsedLP, method: str, M: float = 1e6) -> Tuple[Tableau, Dict[str, int], Dict[str, str]]:
    m = len(p.A)
    n = p.num_vars

    var_names: List[str] = [f"x{i+1}" for i in range(n)]
    var_kind: Dict[str, str] = {f"x{i+1}": "x" for i in range(n)}

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

    tab = [[0.0] * (total_vars + 1) for _ in range(m + 1)]
    basis: List[int] = [0] * m

    for i in range(m):
        for j in range(n):
            tab[i][j] = float(p.A[i][j])
        t = p.constraint_tags[i]
        if p.senses[i] == "<=":
            s = t["slack"]
            tab[i][idx[s]] = 1.0
            basis[i] = idx[s]
        elif p.senses[i] == ">=":
            e = t["excess"]
            a = t["art"]
            tab[i][idx[e]] = -1.0
            tab[i][idx[a]] = 1.0
            basis[i] = idx[a]
        elif p.senses[i] == "=":
            a = t["art"]
            tab[i][idx[a]] = 1.0
            basis[i] = idx[a]

        tab[i][-1] = float(p.b[i])

    obj = tab[m]
    if method == "standard" or method == "bigm":
        for j in range(n):
            obj[j] = -float(p.c[j])
        if method == "bigm":
            for name in var_names:
                if var_kind[name] == "a":
                    obj[idx[name]] = +M
    elif method == "phase1":
        for name in var_names:
            if var_kind[name] == "a":
                obj[idx[name]] = +1.0

    T = Tableau(tab=tab, basis=basis, var_names=var_names)
    _canonicalize_objective(T)

    return T, idx, var_kind


def _remove_columns(T: Tableau, cols_to_remove: List[int]) -> Tableau:
    cols_to_remove = sorted(set(cols_to_remove))
    keep = [j for j in range(T.n()) if j not in cols_to_remove]
    new_names = [T.var_names[j] for j in keep]
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
    phase1_obj = T1.tab[T1.m()][-1]
    if abs(phase1_obj) > 1e-6:
        return "Problema infactible: suma de artificiales > 0 en Fase 1"

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

    # Fase 2
    n_original = p.num_vars
    costs: Dict[str, float] = {f"x{i+1}": float(p.c[i]) for i in range(n_original)}
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
    resultado: Dict[str, float] = {}
    for i in range(p.num_vars):
        name = f"x{i+1}"
        resultado[name] = float(sol.get(name, 0.0))

    for t in p.constraint_tags:
        for k in ("slack", "excess", "art"):
            if k in t:
                nm = t[k]
                resultado[nm] = float(sol.get(nm, 0.0))

    z = objective_value
    if p.is_min:
        z = -z
    if abs(z) < 1e-12:
        z = 0.0
    resultado["Valor Óptimo"] = float(z)
    return resultado


# ======================================================================
# Dualidad
# ======================================================================


def _build_dual_problem(primal: ParsedLP) -> ParsedLP:
    m = len(primal.A)
    n = primal.num_vars
    primal_is_min = primal.is_min

    y_decomp: List[Tuple[str, str, str]] = []
    for i, s in enumerate(primal.senses, start=1):
        if not primal_is_min:
            if s == "<=":
                y_decomp.append(("pos", f"y{i}", ""))
            elif s == ">=":
                y_decomp.append(("neg", f"y{i}p", ""))
            else:
                y_decomp.append(("free", f"y{i}p", f"y{i}n"))
        else:
            if s == ">=":
                y_decomp.append(("pos", f"y{i}", ""))
            elif s == "<=":
                y_decomp.append(("neg", f"y{i}p", ""))
            else:
                y_decomp.append(("free", f"y{i}p", f"y{i}n"))

    dual_is_min = (not primal_is_min)
    dual_c: List[float] = []
    for i in range(m):
        typ, _, _ = y_decomp[i]
        bi = primal.b[i]
        if typ == "pos":
            dual_c.append(bi)
        elif typ == "neg":
            dual_c.append(-bi)
        else:
            dual_c.append(bi)
            dual_c.append(-bi)

    c_original = primal.c if not primal_is_min else [-v for v in primal.c]
    dual_sense = ">=" if not primal_is_min else "<="

    A_dual: List[List[float]] = []
    b_dual: List[float] = []
    senses_dual: List[str] = []
    tags_dual: List[Dict[str, str]] = []

    for j in range(n):
        row: List[float] = []
        for i in range(m):
            aij = primal.A[i][j]
            typ, _, _ = y_decomp[i]
            if typ == "pos":
                row.append(aij)
            elif typ == "neg":
                row.append(-aij)
            else:
                row.append(aij)
                row.append(-aij)
        A_dual.append(row)
        b_dual.append(c_original[j])
        senses_dual.append(dual_sense)
        t: Dict[str, str] = {}
        if dual_sense == "<=":
            t["slack"] = f"s{j+1}"
        else:
            t["excess"] = f"e{j+1}"
            t["art"] = f"a{j+1}"
        tags_dual.append(t)

    return ParsedLP(
        num_vars=len(dual_c),
        is_min=dual_is_min,
        c=(dual_c if not dual_is_min else [-v for v in dual_c]),
        A=A_dual,
        b=b_dual,
        senses=senses_dual,
        constraint_tags=tags_dual,
    )


def _reconstruct_dual_values(dual: ParsedLP, dual_sol: Dict[str, float]) -> Dict[str, float]:
    duals: Dict[str, float] = {}
    for i in range(dual.num_vars):
        duals[f"y{i+1}"] = float(dual_sol.get(f"x{i+1}", 0.0))
    return duals


# ======================================================================
# Sensibilidad (extracción numérica)
# ======================================================================


def _infer_num_vars(funcion_obj: str, restricciones: List[str]) -> int:
    max_idx = 0
    text = (funcion_obj or "") + " " + " ".join(restricciones or [])
    for m in re.finditer(r"x(\d+)", text):
        try:
            max_idx = max(max_idx, int(m.group(1)))
        except Exception:
            continue
    return max_idx


def _solve_for_sensitivity(p: ParsedLP):
    if all(s == "<=" for s in p.senses):
        T, _, _ = _build_initial_tableau(p, method="standard")
        status, T = _simplex(T)
        if status != "Óptimo":
            return f"Problema {status}"
        sol = _extract_solution(T)
        out = _format_result(p, sol, objective_value=T.tab[T.m()][-1])
        return out, T
    
    # Dos fases devolviendo Tableau
    T1, _, kind = _build_initial_tableau(p, method="phase1")
    status1, T1 = _simplex(T1)
    if status1 != "Óptimo":
        return f"Fase 1: {status1}"
    if abs(T1.tab[T1.m()][-1]) > 1e-6:
        return "Problema infactible: suma de artificiales > 0 en Fase 1"

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

    n_original = p.num_vars
    costs: Dict[str, float] = {f"x{i+1}": float(p.c[i]) for i in range(n_original)}
    obj = T2.tab[T2.m()]
    for j, name in enumerate(T2.var_names):
        obj[j] = -costs.get(name, 0.0)
    obj[-1] = 0.0
    _canonicalize_objective(T2)
    status2, T2 = _simplex(T2)
    if status2 != "Óptimo":
        return f"Fase 2: {status2}"
    sol = _extract_solution(T2)
    out = _format_result(p, sol, objective_value=T2.tab[T2.m()][-1])
    return out, T2


def _compute_constraint_slacks(constraints: List[str], x_vals: Dict[str, float], num_vars: int) -> List[Dict[str, float]]:
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
    m = len(p.A)
    if T.m() != m:
        return [None] * m

    c_original = p.c if not p.is_min else [-v for v in p.c]
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
        col_vectors[name] = vec

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

    BT = [[B[j][i] for j in range(m)] for i in range(m)]
    y = _solve_linear_system(BT, cB)
    if y is None:
        return [None] * m
    return [float(v) for v in y]


def _compute_reduced_costs(p: ParsedLP, T: Tableau, shadow_prices: List[Optional[float]]) -> Dict[str, float]:
    m = len(p.A)
    c_original = p.c if not p.is_min else [-v for v in p.c]
    if any(v is None for v in shadow_prices) or len(shadow_prices) != m:
        out = {}
        for name in T.var_names:
            if name.startswith("x") or name.startswith("s") or name.startswith("e"):
                out[name] = 0.0
        return out

    y = [float(v) for v in shadow_prices]

    def dot(u, v):
        s = 0.0
        for a, b in zip(u, v):
            s += a * b
        return s

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

    for bcol in T.basis:
        bname = T.var_names[bcol]
        if bname in out:
            out[bname] = 0.0
    return out


def _solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        return None
    M = [row[:] + [float(b[i])] for i, row in enumerate(A)]
    for col in range(n):
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
        pv = M[col][col]
        inv = 1.0 / pv
        for j in range(col, n + 1):
            M[col][j] *= inv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    return [float(M[i][n]) for i in range(n)]
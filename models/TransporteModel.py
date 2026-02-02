"""Modelo de Transporte (Python puro).

Implementa los métodos exigidos por la guía (sin NumPy ni librerías de optimización):
  - Esquina Noroeste
  - Costo Mínimo
  - Aproximación de Vogel
  - Prueba de optimalidad y determinación de la solución óptima (MODI / u-v)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from utils.gemini_client import get_api_key
from utils.sensitivity_ai import generate_sensitivity_report


_TOL = 1e-9


def _deepcopy_matrix(m: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in m]


def _sum_list(xs: List[float]) -> float:
    s = 0.0
    for v in xs:
        s += float(v)
    return s


def _balance_problem(
    oferta: List[float],
    demanda: List[float],
    costos: List[List[float]],
) -> Tuple[List[float], List[float], List[List[float]]]:
    """Balancea el problema agregando fila/columna dummy con costo 0 si hace falta."""

    oferta_b = [float(x) for x in oferta]
    demanda_b = [float(x) for x in demanda]
    costos_b = [[float(c) for c in row] for row in costos]

    so = _sum_list(oferta_b)
    sd = _sum_list(demanda_b)

    if abs(so - sd) <= _TOL:
        return oferta_b, demanda_b, costos_b

    if so < sd:
        # Agregar oferta dummy (fila)
        oferta_b.append(sd - so)
        n = len(demanda_b)
        costos_b.append([0.0] * n)
    else:
        # Agregar demanda dummy (columna)
        demanda_b.append(so - sd)
        for row in costos_b:
            row.append(0.0)

    return oferta_b, demanda_b, costos_b


def _costo_total(asignacion: List[List[float]], costos: List[List[float]]) -> float:
    total = 0.0
    for i in range(len(asignacion)):
        for j in range(len(asignacion[0])):
            total += float(asignacion[i][j]) * float(costos[i][j])
    return total


def _basic_cells(asignacion: List[List[float]]) -> Set[Tuple[int, int]]:
    basics: Set[Tuple[int, int]] = set()
    m = len(asignacion)
    n = len(asignacion[0])
    for i in range(m):
        for j in range(n):
            if asignacion[i][j] > _TOL:
                basics.add((i, j))
    return basics


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def _complete_basis(
    asignacion: List[List[float]],
    costos: List[List[float]],
    basics: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    m = len(asignacion)
    n = len(asignacion[0])
    target = m + n - 1

    dsu = _DSU(m + n)
    for (i, j) in basics:
        dsu.union(i, m + j)

    if len(basics) >= target:
        return basics

    candidates: List[Tuple[float, int, int]] = []
    for i in range(m):
        for j in range(n):
            if (i, j) not in basics:
                candidates.append((float(costos[i][j]), i, j))
    candidates.sort(key=lambda x: x[0])

    for _, i, j in candidates:
        if len(basics) >= target:
            break
        if dsu.union(i, m + j):
            basics.add((i, j))

    if len(basics) < target:
        for _, i, j in candidates:
            if len(basics) >= target:
                break
            basics.add((i, j))

    return basics


def _compute_potentials(
    basics: Set[Tuple[int, int]],
    costos: List[List[float]],
    m: int,
    n: int,
) -> Tuple[List[float], List[float]]:
    u: List[Optional[float]] = [None] * m
    v: List[Optional[float]] = [None] * n
    u[0] = 0.0

    changed = True
    while changed:
        changed = False
        for (i, j) in basics:
            cij = float(costos[i][j])
            if u[i] is not None and v[j] is None:
                v[j] = cij - u[i]
                changed = True
            elif u[i] is None and v[j] is not None:
                u[i] = cij - v[j]
                changed = True

        if not changed and (None in u or None in v):
            for i in range(m):
                if u[i] is None:
                    u[i] = 0.0
                    changed = True
                    break

    u_out = [float(x) if x is not None else 0.0 for x in u]
    v_out = [float(x) if x is not None else 0.0 for x in v]
    return u_out, v_out


def _reduced_costs(
    costos: List[List[float]],
    u: List[float],
    v: List[float],
) -> List[List[float]]:
    m = len(costos)
    n = len(costos[0])
    rc = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            rc[i][j] = float(costos[i][j]) - (float(u[i]) + float(v[j]))
    return rc


def _build_basis_graph(
    basics: Set[Tuple[int, int]],
    m: int,
    n: int,
) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
    g: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}

    def add_edge(a: Tuple[str, int], b: Tuple[str, int]) -> None:
        g.setdefault(a, []).append(b)
        g.setdefault(b, []).append(a)

    for (i, j) in basics:
        add_edge(("r", i), ("c", j))

    for i in range(m):
        g.setdefault(("r", i), [])
    for j in range(n):
        g.setdefault(("c", j), [])

    return g


def _find_path(
    g: Dict[Tuple[str, int], List[Tuple[str, int]]],
    start: Tuple[str, int],
    goal: Tuple[str, int],
) -> List[Tuple[str, int]]:
    from collections import deque

    q = deque([start])
    parent: Dict[Tuple[str, int], Optional[Tuple[str, int]]] = {start: None}

    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in g.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)

    if goal not in parent:
        return []

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _cycle_cells_from_path(
    path_nodes: List[Tuple[str, int]],
    entering: Tuple[int, int],
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = [entering]
    for a, b in zip(path_nodes, path_nodes[1:]):
        if a[0] == "r" and b[0] == "c":
            cells.append((a[1], b[1]))
        elif a[0] == "c" and b[0] == "r":
            cells.append((b[1], a[1]))
        else:
            continue
    return cells


def _improve_modi(
    asignacion: List[List[float]],
    costos: List[List[float]],
    max_iters: int = 200,
) -> Dict[str, object]:
    m = len(asignacion)
    n = len(asignacion[0])
    alloc = _deepcopy_matrix(asignacion)

    basics = _basic_cells(alloc)
    basics = _complete_basis(alloc, costos, basics)

    history: List[Dict[str, object]] = []

    for _it in range(max_iters):
        u, v = _compute_potentials(basics, costos, m, n)
        rc = _reduced_costs(costos, u, v)

        best = (0.0, -1, -1)
        for i in range(m):
            for j in range(n):
                if (i, j) in basics:
                    continue
                if rc[i][j] < best[0] - _TOL:
                    best = (rc[i][j], i, j)

        if best[1] == -1:
            return {
                "asignacion": alloc,
                "u": u,
                "v": v,
                "costos_reducidos": rc,
                "basicas": sorted(list(basics)),
                "iteraciones": history,
                "optimo": True,
            }

        entering = (best[1], best[2])
        g = _build_basis_graph(basics, m, n)
        path = _find_path(g, ("r", entering[0]), ("c", entering[1]))
        if not path:
            basics.add(entering)
            basics = _complete_basis(alloc, costos, basics)
            continue

        cycle_cells = _cycle_cells_from_path(path, entering)
        plus_cells = []
        minus_cells = []
        for idx, cell in enumerate(cycle_cells):
            if idx % 2 == 0:
                plus_cells.append(cell)
            else:
                minus_cells.append(cell)

        theta = None
        leaving = None
        for (i, j) in minus_cells:
            val = alloc[i][j]
            if theta is None or val < theta - _TOL:
                theta = val
                leaving = (i, j)

        if theta is None:
            return {
                "asignacion": alloc,
                "u": u,
                "v": v,
                "costos_reducidos": rc,
                "basicas": sorted(list(basics)),
                "iteraciones": history,
                "optimo": False,
                "error": "No se pudo construir el ciclo MODI.",
            }

        for (i, j) in plus_cells:
            alloc[i][j] += theta
        for (i, j) in minus_cells:
            alloc[i][j] -= theta
            if abs(alloc[i][j]) <= _TOL:
                alloc[i][j] = 0.0

        basics.add(entering)
        if leaving in basics:
            basics.remove(leaving)
        basics = _complete_basis(alloc, costos, basics)

        history.append(
            {
                "entrada": entering,
                "sale": leaving,
                "theta": theta,
                "rc_entrada": best[0],
            }
        )

    u, v = _compute_potentials(basics, costos, m, n)
    rc = _reduced_costs(costos, u, v)
    return {
        "asignacion": alloc,
        "u": u,
        "v": v,
        "costos_reducidos": rc,
        "basicas": sorted(list(basics)),
        "iteraciones": history,
        "optimo": False,
        "error": "Se alcanzó el máximo de iteraciones en MODI.",
    }


class TransporteModel:
    def __init__(self):
        self.api_key = get_api_key()

    def esquina_noroeste(self, datos: Dict) -> Dict:
        oferta_b, demanda_b, costos_b = _balance_problem(datos["oferta"], datos["demanda"], datos["costos"])
        oferta = oferta_b[:]
        demanda = demanda_b[:]
        m = len(oferta)
        n = len(demanda)

        asignacion = [[0.0 for _ in range(n)] for _ in range(m)]
        i, j = 0, 0
        while i < m and j < n:
            cantidad = min(oferta[i], demanda[j])
            asignacion[i][j] = cantidad
            oferta[i] -= cantidad
            demanda[j] -= cantidad
            if oferta[i] <= _TOL:
                i += 1
            if j < n and demanda[j] <= _TOL:
                j += 1

        costo_total = _costo_total(asignacion, costos_b)
        return {
            "metodo": "Esquina Noroeste",
            "asignacion": [[int(x) if abs(x - round(x)) <= _TOL else x for x in row] for row in asignacion],
            "costo_total": costo_total,
            "oferta_bal": oferta_b,
            "demanda_bal": demanda_b,
            "costos_bal": costos_b,
        }

    def costo_minimo(self, datos: Dict) -> Dict:
        oferta_b, demanda_b, costos_b = _balance_problem(datos["oferta"], datos["demanda"], datos["costos"])
        oferta = oferta_b[:]
        demanda = demanda_b[:]
        m = len(oferta)
        n = len(demanda)

        asignacion = [[0.0 for _ in range(n)] for _ in range(m)]

        while True:
            min_cost = None
            min_i, min_j = -1, -1
            for i in range(m):
                if oferta[i] <= _TOL:
                    continue
                for j in range(n):
                    if demanda[j] <= _TOL:
                        continue
                    c = float(costos_b[i][j])
                    if min_cost is None or c < min_cost - _TOL:
                        min_cost, min_i, min_j = c, i, j

            if min_i == -1:
                break

            cantidad = min(oferta[min_i], demanda[min_j])
            asignacion[min_i][min_j] = cantidad
            oferta[min_i] -= cantidad
            demanda[min_j] -= cantidad

        costo_total = _costo_total(asignacion, costos_b)
        return {
            "metodo": "Costo Mínimo",
            "asignacion": [[int(x) if abs(x - round(x)) <= _TOL else x for x in row] for row in asignacion],
            "costo_total": costo_total,
            "oferta_bal": oferta_b,
            "demanda_bal": demanda_b,
            "costos_bal": costos_b,
        }

    def vogel(self, datos: Dict) -> Dict:
        oferta_b, demanda_b, costos_b = _balance_problem(datos["oferta"], datos["demanda"], datos["costos"])
        oferta = oferta_b[:]
        demanda = demanda_b[:]
        m = len(oferta)
        n = len(demanda)

        asignacion = [[0.0 for _ in range(n)] for _ in range(m)]
        filas_activas = set(range(m))
        cols_activas = set(range(n))

        while filas_activas and cols_activas:
            pen_f: Dict[int, float] = {}
            for i in list(filas_activas):
                if oferta[i] <= _TOL:
                    filas_activas.discard(i)
                    continue
                costos_validos = [float(costos_b[i][j]) for j in cols_activas if demanda[j] > _TOL]
                if len(costos_validos) == 0:
                    filas_activas.discard(i)
                    continue
                costos_validos.sort()
                pen_f[i] = (costos_validos[1] - costos_validos[0]) if len(costos_validos) > 1 else 0.0

            pen_c: Dict[int, float] = {}
            for j in list(cols_activas):
                if demanda[j] <= _TOL:
                    cols_activas.discard(j)
                    continue
                costos_validos = [float(costos_b[i][j]) for i in filas_activas if oferta[i] > _TOL]
                if len(costos_validos) == 0:
                    cols_activas.discard(j)
                    continue
                costos_validos.sort()
                pen_c[j] = (costos_validos[1] - costos_validos[0]) if len(costos_validos) > 1 else 0.0

            if not pen_f and not pen_c:
                break

            max_f = max(pen_f.items(), key=lambda kv: kv[1]) if pen_f else (None, -1.0)
            max_c = max(pen_c.items(), key=lambda kv: kv[1]) if pen_c else (None, -1.0)

            if max_f[1] >= max_c[1]:
                i = int(max_f[0])
                best_j = None
                best_cost = None
                for j in cols_activas:
                    if demanda[j] <= _TOL:
                        continue
                    c = float(costos_b[i][j])
                    if best_cost is None or c < best_cost - _TOL:
                        best_cost, best_j = c, j
                j = int(best_j)
            else:
                j = int(max_c[0])
                best_i = None
                best_cost = None
                for i in filas_activas:
                    if oferta[i] <= _TOL:
                        continue
                    c = float(costos_b[i][j])
                    if best_cost is None or c < best_cost - _TOL:
                        best_cost, best_i = c, i
                i = int(best_i)

            cantidad = min(oferta[i], demanda[j])
            asignacion[i][j] = cantidad
            oferta[i] -= cantidad
            demanda[j] -= cantidad

            if oferta[i] <= _TOL:
                filas_activas.discard(i)
            if demanda[j] <= _TOL:
                cols_activas.discard(j)

        costo_total = _costo_total(asignacion, costos_b)
        return {
            "metodo": "Vogel",
            "asignacion": [[int(x) if abs(x - round(x)) <= _TOL else x for x in row] for row in asignacion],
            "costo_total": costo_total,
            "oferta_bal": oferta_b,
            "demanda_bal": demanda_b,
            "costos_bal": costos_b,
        }

    def prueba_optimalidad(self, datos: Dict, solucion_inicial: Dict) -> Dict:
        if solucion_inicial is None or "asignacion" not in solucion_inicial:
            return {"error": "No se proporcionó una solución inicial."}

        if all(k in solucion_inicial for k in ("oferta_bal", "demanda_bal", "costos_bal")):
            oferta_b = [float(x) for x in solucion_inicial["oferta_bal"]]
            demanda_b = [float(x) for x in solucion_inicial["demanda_bal"]]
            costos_b = [[float(c) for c in row] for row in solucion_inicial["costos_bal"]]
        else:
            oferta_b, demanda_b, costos_b = _balance_problem(datos["oferta"], datos["demanda"], datos["costos"])

        asignacion = [[float(x) for x in row] for row in solucion_inicial["asignacion"]]
        res = _improve_modi(asignacion, costos_b)
        asign_opt = res["asignacion"]
        costo_total = _costo_total(asign_opt, costos_b)

        mensaje = "La solución es óptima." if res.get("optimo", False) else "Se encontró una solución mejorada (puede no ser óptima)."
        if "error" in res:
            mensaje = f"{mensaje} Nota: {res['error']}"

        asign_fmt = [[int(x) if abs(x - round(x)) <= _TOL else x for x in row] for row in asign_opt]

        return {
            "metodo": "Prueba de Optimalidad",
            "mensaje": mensaje,
            "asignacion": asign_fmt,
            "costo_total": costo_total,
            "u": res.get("u"),
            "v": res.get("v"),
            "costos_reducidos": res.get("costos_reducidos"),
            "basicas": res.get("basicas"),
            "iteraciones": res.get("iteraciones"),
            "oferta_bal": oferta_b,
            "demanda_bal": demanda_b,
            "costos_bal": costos_b,
        }

    def construir_contexto_sensibilidad(self, resultado: Dict, datos: Optional[Dict] = None) -> Dict:
        if resultado is None or "asignacion" not in resultado:
            return {"error": "No hay resultado para analizar."}

        if all(k in resultado for k in ("oferta_bal", "demanda_bal", "costos_bal")):
            oferta_b = [float(x) for x in resultado["oferta_bal"]]
            demanda_b = [float(x) for x in resultado["demanda_bal"]]
            costos_b = [[float(c) for c in row] for row in resultado["costos_bal"]]
        elif datos is not None:
            oferta_b, demanda_b, costos_b = _balance_problem(datos["oferta"], datos["demanda"], datos["costos"])
        else:
            return {"error": "No hay datos suficientes para sensibilidad."}

        asign = [[float(x) for x in row] for row in resultado["asignacion"]]
        m = len(costos_b)
        n = len(costos_b[0])
        
        basics = _basic_cells(asign)
        basics = _complete_basis(asign, costos_b, basics)
        u, v = _compute_potentials(basics, costos_b, m, n)
        rc = _reduced_costs(costos_b, u, v)
        total = _costo_total(asign, costos_b)

        tol = 1e-9
        violations = []
        for i in range(m):
            for j in range(n):
                if (i, j) not in basics and rc[i][j] < -tol:
                    violations.append({"cell": [i, j], "rc": float(rc[i][j]), "c": float(costos_b[i][j])})

        facts = {
            "module": "transporte",
            "metodo": resultado.get("metodo"),
            "total_cost": float(total),
            "oferta": oferta_b,
            "demanda": demanda_b,
            "costos": costos_b,
            "asignacion": asign,
            "u": u,
            "v": v,
            "reduced_costs": rc,
            "basicas": sorted([list(x) for x in basics]),
            "violations": violations,
            "is_optimal": (len(violations) == 0),
            "nota": "Minimización: óptimo si rc >= 0 en celdas NO básicas (MODI).",
        }
        return facts

    def analizar_sensibilidad(self, resultado: Dict, datos: Optional[Dict] = None) -> str:
        """Análisis de sensibilidad con guardrails e inyección de contexto."""
        try:
            facts = self.construir_contexto_sensibilidad(resultado, datos)
            if "error" in facts:
                return str(facts["error"])
            
            # --- NUEVO: Extraer contexto ---
            ctx = ""
            if datos:
                ctx = datos.get("contexto", "")
            # -------------------------------

            return generate_sensitivity_report(
                'transporte', 
                facts, 
                api_key=self.api_key, 
                max_retries=1, 
                context=ctx  # Pasar el contexto
            )
        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"
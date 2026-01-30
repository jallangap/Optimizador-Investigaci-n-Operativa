import math
import heapq
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

from utils.gemini_client import generate_text
from utils.sensitivity_ai import generate_sensitivity_report
from utils.sensitivity_context import set_last_facts


class _Edge:
    """Arista para grafos residuales (flujo máximo / costo mínimo)."""

    __slots__ = ("to", "rev", "cap", "cost", "orig_cap")

    def __init__(self, to: str, rev: int, cap: int, cost: int = 0):
        self.to = to
        self.rev = rev
        self.cap = int(cap)
        self.cost = int(cost)
        self.orig_cap = int(cap)


class _ResidualGraph:
    def __init__(self):
        self.adj: Dict[str, List[_Edge]] = {}

    def _ensure(self, u: str) -> None:
        if u not in self.adj:
            self.adj[u] = []

    def add_edge(self, u: str, v: str, cap: int, cost: int = 0) -> _Edge:
        """Agrega arista dirigida u->v con capacidad y costo.

        Crea también la arista reversa (cap=0, cost=-cost).
        Retorna la referencia a la arista forward para reconstruir el flujo final.
        """
        self._ensure(u)
        self._ensure(v)
        fwd = _Edge(v, len(self.adj[v]), cap, cost)
        rev = _Edge(u, len(self.adj[u]), 0, -cost)
        self.adj[u].append(fwd)
        self.adj[v].append(rev)
        return fwd


class _UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


class RedesModel:
    """Modelo de Redes: algoritmos 100% manuales (Python puro)."""

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

    # ------------------------
    # Utilidades de parsing
    # ------------------------
    @staticmethod
    def _clean_token(s: str) -> str:
        return s.strip()

    def _parse_undirected_weighted(self, aristas: List[str]) -> Tuple[Optional[List[Tuple[str, str, int]]], Optional[str]]:
        parsed: List[Tuple[str, str, int]] = []
        for a in aristas:
            a = a.strip()
            if not a:
                continue
            parts = [self._clean_token(p) for p in a.split("-")]
            if len(parts) != 3:
                return None, f"Formato de arista incorrecto: '{a}'. Usa Origen-Destino-Peso (ej: A-B-2)"
            u, v, w = parts
            try:
                w_i = int(w)
            except ValueError:
                return None, f"Peso no válido en arista '{a}'. Debe ser entero."
            parsed.append((u, v, w_i))
        if not parsed:
            return None, "No se encontraron aristas válidas."
        return parsed, None

    def _parse_directed_capacity(self, aristas: List[str]) -> Tuple[Optional[List[Tuple[str, str, int]]], Optional[str]]:
        parsed: List[Tuple[str, str, int]] = []
        for a in aristas:
            a = a.strip()
            if not a:
                continue
            parts = [self._clean_token(p) for p in a.split("-")]
            if len(parts) != 3:
                return None, f"Formato de arista incorrecto: '{a}'. Usa Origen-Destino-Capacidad (ej: A-B-10)"
            u, v, c = parts
            try:
                c_i = int(c)
            except ValueError:
                return None, f"Capacidad no válida en arista '{a}'. Debe ser entero."
            if c_i < 0:
                return None, f"Capacidad negativa en arista '{a}'."
            parsed.append((u, v, c_i))
        if not parsed:
            return None, "No se encontraron aristas válidas."
        return parsed, None

    def _parse_directed_capacity_cost(self, aristas: List[str]) -> Tuple[Optional[List[Tuple[str, str, int, int]]], Optional[str]]:
        parsed: List[Tuple[str, str, int, int]] = []
        for a in aristas:
            a = a.strip()
            if not a:
                continue
            parts = [self._clean_token(p) for p in a.split("-")]
            if len(parts) != 4:
                return None, (
                    f"Formato de arista incorrecto: '{a}'. Para Flujo de Costo Mínimo usa "
                    f"Origen-Destino-Capacidad-Costo (ej: A-B-10-3)"
                )
            u, v, cap, cost = parts
            try:
                cap_i = int(cap)
                cost_i = int(cost)
            except ValueError:
                return None, f"Capacidad/Costo no válido en arista '{a}'. Deben ser enteros."
            if cap_i < 0:
                return None, f"Capacidad negativa en arista '{a}'."
            parsed.append((u, v, cap_i, cost_i))
        if not parsed:
            return None, "No se encontraron aristas válidas."
        return parsed, None

    # ------------------------
    # Gráfico (sin networkx)
    # ------------------------
    @staticmethod
    def _circle_layout(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
        n = max(len(nodes), 1)
        pos: Dict[str, Tuple[float, float]] = {}
        for i, node in enumerate(nodes):
            theta = 2.0 * math.pi * i / n
            pos[node] = (math.cos(theta), math.sin(theta))
        return pos

    def crear_grafico(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        edge_labels: Optional[Dict[Tuple[str, str], str]] = None,
        directed: bool = False,
        highlight_path: Optional[List[str]] = None,
    ) -> None:
        """Genera una imagen simple 'red.png' sin depender de NetworkX."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return  # Si no hay matplotlib, no interrumpimos la app

        pos = self._circle_layout(nodes)

        plt.figure(figsize=(6, 4))
        # Nodos
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        plt.scatter(xs, ys)
        for n in nodes:
            x, y = pos[n]
            plt.text(x, y, n, ha="center", va="center")

        # Resaltar ruta si aplica
        highlight_set = set()
        if highlight_path and len(highlight_path) >= 2:
            for i in range(len(highlight_path) - 1):
                highlight_set.add((highlight_path[i], highlight_path[i + 1]))
                highlight_set.add((highlight_path[i + 1], highlight_path[i]))  # por si es no dirigido

        # Aristas
        for (u, v) in edges:
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            is_high = (u, v) in highlight_set
            # No fijamos colores por estilo; pero para resaltar sin color, aumentamos grosor
            lw = 2.5 if is_high else 1.0

            if directed:
                # Flecha simple
                plt.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw),
                )
            else:
                plt.plot([x1, x2], [y1, y2], linewidth=lw)

            if edge_labels:
                label = edge_labels.get((u, v))
                if label is None and not directed:
                    # En no dirigidos, probar (v,u)
                    label = edge_labels.get((v, u))
                if label:
                    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    plt.text(mx, my, label, ha="center", va="center")

        plt.axis("off")
        plt.tight_layout()
        plt.savefig("red.png")
        plt.close()

    # ------------------------
    # Algoritmos
    # ------------------------
    def ruta_mas_corta(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        edges, err = self._parse_undirected_weighted(datos.get("aristas", []))
        if err:
            return {"error": err}

        nodes = [n.strip() for n in datos.get("nodos", []) if n.strip()]
        if not nodes:
            return {"error": "No se proporcionaron nodos válidos."}
        s, t = nodes[0], nodes[-1]

        # Adyacencia
        adj: Dict[str, List[Tuple[str, int]]] = {}
        for u, v, w in edges:
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

        if s not in adj or t not in adj:
            return {"error": "Nodo origen/destino no existe en las aristas."}

        # Dijkstra
        INF = 10 ** 18
        dist: Dict[str, int] = {n: INF for n in adj.keys()}
        prev: Dict[str, Optional[str]] = {n: None for n in adj.keys()}
        dist[s] = 0
        pq: List[Tuple[int, str]] = [(0, s)]

        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            if u == t:
                break
            for v, w in adj.get(u, []):
                if w < 0:
                    return {"error": "Dijkstra requiere pesos no negativos. Detectado peso negativo."}
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist.get(t, INF) >= INF:
            return {"error": "No hay ruta entre los nodos especificados."}

        # Reconstrucción de ruta
        ruta: List[str] = []
        cur = t
        while cur is not None:
            ruta.append(cur)
            cur = prev.get(cur)
        ruta.reverse()

        # Gráfico
        edge_list = [(u, v) for (u, v, _) in edges]
        labels = {(u, v): str(w) for (u, v, w) in edges}
        self.crear_grafico(nodes=list({*nodes, *[u for u, _, _ in edges], *[v for _, v, _ in edges]}),
                           edges=edge_list,
                           edge_labels=labels,
                           directed=False,
                           highlight_path=ruta)

        return {
            "metodo": "Ruta Más Corta",
            "ruta": ruta,
            "distancia": dist[t],
        }

    def arbol_minima_expansion(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        edges, err = self._parse_undirected_weighted(datos.get("aristas", []))
        if err:
            return {"error": err}

        nodes = [n.strip() for n in datos.get("nodos", []) if n.strip()]
        # Si el usuario no listó todos los nodos, los inferimos de las aristas
        node_set = set(nodes)
        for u, v, _ in edges:
            node_set.add(u)
            node_set.add(v)
        nodes = list(node_set)

        uf = _UnionFind(nodes)
        edges_sorted = sorted(edges, key=lambda x: x[2])
        mst: List[Tuple[str, str, int]] = []
        total = 0

        for u, v, w in edges_sorted:
            if uf.union(u, v):
                mst.append((u, v, w))
                total += w

        # Validación de conectividad (n-1 aristas)
        if len(nodes) > 0 and len(mst) != len(nodes) - 1:
            return {"error": "El grafo no es conexo: no se puede construir un árbol de expansión mínima."}

        edge_list = [(u, v) for (u, v, _) in edges]
        labels = {(u, v): str(w) for (u, v, w) in edges}
        self.crear_grafico(nodes=nodes, edges=edge_list, edge_labels=labels, directed=False)

        return {
            "metodo": "Árbol de Mínima Expansión",
            "arbol": [(u, v) for (u, v, _) in mst],
            "costo_total": total,
            "arbol_detalle": mst,
        }

    def flujo_maximo(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        parsed, err = self._parse_directed_capacity(datos.get("aristas", []))
        if err:
            return {"error": err}

        nodes = [n.strip() for n in datos.get("nodos", []) if n.strip()]
        if not nodes:
            return {"error": "No se proporcionaron nodos válidos."}
        s, t = nodes[0], nodes[-1]

        g = _ResidualGraph()
        original_edges: List[Tuple[str, str, _Edge]] = []
        for u, v, cap in parsed:
            fwd = g.add_edge(u, v, cap, 0)
            original_edges.append((u, v, fwd))

        if s not in g.adj or t not in g.adj:
            return {"error": "Nodo origen/destino no existe en las aristas."}

        flow = 0
        while True:
            parent: Dict[str, Tuple[Optional[str], Optional[int]]] = {s: (None, None)}
            q = deque([s])
            while q and t not in parent:
                u = q.popleft()
                for idx, e in enumerate(g.adj[u]):
                    if e.cap > 0 and e.to not in parent:
                        parent[e.to] = (u, idx)
                        q.append(e.to)
                        if e.to == t:
                            break

            if t not in parent:
                break

            # Bottleneck
            add = 10 ** 18
            v = t
            while v != s:
                pu, pidx = parent[v]
                assert pu is not None and pidx is not None
                e = g.adj[pu][pidx]
                add = min(add, e.cap)
                v = pu

            # Augment
            v = t
            while v != s:
                pu, pidx = parent[v]
                assert pu is not None and pidx is not None
                e = g.adj[pu][pidx]
                e.cap -= add
                g.adj[e.to][e.rev].cap += add
                v = pu

            flow += add

        # Construir flujo_dict estilo networkx
        flujo_dict: Dict[str, Dict[str, int]] = {}
        for u, v, e in original_edges:
            f = e.orig_cap - e.cap
            flujo_dict.setdefault(u, {})[v] = f

        # Gráfico
        edge_list = [(u, v) for (u, v, _) in parsed]
        labels = {(u, v): str(cap) for (u, v, cap) in parsed}
        all_nodes = list({*nodes, *[u for u, _, _ in parsed], *[v for _, v, _ in parsed]})
        self.crear_grafico(nodes=all_nodes, edges=edge_list, edge_labels=labels, directed=True)

        return {"metodo": "Flujo Máximo", "flujo_valor": flow, "flujo_dict": flujo_dict}

    def flujo_costo_minimo(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Min-Cost Max-Flow (flujo máximo con costo mínimo) desde nodo[0] hasta nodo[-1]."""
        parsed, err = self._parse_directed_capacity_cost(datos.get("aristas", []))
        if err:
            return {"error": err}

        nodes = [n.strip() for n in datos.get("nodos", []) if n.strip()]
        if not nodes:
            return {"error": "No se proporcionaron nodos válidos."}
        s, t = nodes[0], nodes[-1]

        g = _ResidualGraph()
        original_edges: List[Tuple[str, str, _Edge, int]] = []  # (u,v,edge,cost)
        node_set = set(nodes)
        for u, v, cap, cost in parsed:
            node_set.add(u)
            node_set.add(v)
            fwd = g.add_edge(u, v, cap, cost)
            original_edges.append((u, v, fwd, cost))

        if s not in g.adj or t not in g.adj:
            return {"error": "Nodo origen/destino no existe en las aristas."}

        flow = 0
        total_cost = 0

        # Successive Shortest Augmenting Path usando SPFA (Bellman-Ford con cola)
        while True:
            dist: Dict[str, int] = {n: 10 ** 18 for n in g.adj.keys()}
            inq: Dict[str, bool] = {n: False for n in g.adj.keys()}
            prev_node: Dict[str, Optional[str]] = {n: None for n in g.adj.keys()}
            prev_idx: Dict[str, Optional[int]] = {n: None for n in g.adj.keys()}

            dist[s] = 0
            q = deque([s])
            inq[s] = True

            while q:
                u = q.popleft()
                inq[u] = False
                for idx, e in enumerate(g.adj[u]):
                    if e.cap <= 0:
                        continue
                    nd = dist[u] + e.cost
                    if nd < dist.get(e.to, 10 ** 18):
                        dist[e.to] = nd
                        prev_node[e.to] = u
                        prev_idx[e.to] = idx
                        if not inq[e.to]:
                            q.append(e.to)
                            inq[e.to] = True

            if prev_node[t] is None:
                break  # no hay más camino augmentante

            # Bottleneck
            add = 10 ** 18
            v = t
            while v != s:
                u = prev_node[v]
                idx = prev_idx[v]
                assert u is not None and idx is not None
                e = g.adj[u][idx]
                add = min(add, e.cap)
                v = u

            # Augment
            v = t
            while v != s:
                u = prev_node[v]
                idx = prev_idx[v]
                assert u is not None and idx is not None
                e = g.adj[u][idx]
                e.cap -= add
                g.adj[e.to][e.rev].cap += add
                total_cost += add * e.cost
                v = u

            flow += add

        flujo_dict: Dict[str, Dict[str, int]] = {}
        for u, v, e, _cost in original_edges:
            f = e.orig_cap - e.cap
            flujo_dict.setdefault(u, {})[v] = f

        # Gráfico
        edge_list = [(u, v) for (u, v, _, _) in parsed]
        labels = {(u, v): f"{cap}/{cost}" for (u, v, cap, cost) in parsed}
        self.crear_grafico(nodes=list(node_set), edges=edge_list, edge_labels=labels, directed=True)

        return {
            "metodo": "Flujo de Costo Mínimo",
            "flujo_valor": flow,
            "costo_total": total_cost,
            "flujo_dict": flujo_dict,
        }

    # ------------------------
    # Sensibilidad (IA) - con guardrails
    # ------------------------
    def construir_contexto_sensibilidad(self, resultado: Dict[str, Any], datos: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construye un JSON de hechos (sin IA) para sensibilidad.

        NOTA:
        - Para evitar alucinaciones, se incluyen entradas (nodos/aristas) y salidas reales.
        - Se agregan evidencias cuando existen (p.ej., aristas saturadas, min-cut en flujo máximo).
        """
        if resultado is None or isinstance(resultado, str) or 'error' in resultado:
            return {"error": "No hay un resultado válido para analizar."}

        metodo = resultado.get('metodo')
        facts: Dict[str, Any] = {"module": "redes", "metodo": metodo}

        # Si no hay datos de entrada, igual mandamos el resultado (pero será más limitado)
        nodes_in = []
        edges_in = []
        if datos:
            nodes_in = [str(n).strip() for n in (datos.get('nodos') or []) if str(n).strip()]
            edges_in = [str(a).strip() for a in (datos.get('aristas') or []) if str(a).strip()]
            facts["nodos"] = nodes_in
            facts["aristas_raw"] = edges_in

        if metodo == 'Ruta Más Corta':
            facts["ruta"] = resultado.get('ruta')
            facts["distancia"] = resultado.get('distancia')
            # Evidencia: aristas usadas con pesos
            if datos and edges_in:
                parsed, err = self._parse_undirected_weighted(edges_in)
                if not err and parsed is not None:
                    wmap = {}
                    for u, v, w in parsed:
                        wmap[(u, v)] = w
                        wmap[(v, u)] = w
                    ruta = resultado.get('ruta') or []
                    used = []
                    for i in range(len(ruta) - 1):
                        u, v = ruta[i], ruta[i+1]
                        if (u, v) in wmap:
                            used.append({"u": u, "v": v, "peso": wmap[(u, v)]})
                    facts["used_edges"] = used
            return facts

        if metodo == 'Árbol de Mínima Expansión':
            facts["arbol"] = resultado.get('arbol')
            facts["costo_total"] = resultado.get('costo_total')
            return facts

        if metodo == 'Flujo Máximo':
            facts["flow_value"] = resultado.get('flujo_valor')
            facts["flow_dict"] = resultado.get('flujo_dict')
            if datos and edges_in:
                parsed, err = self._parse_directed_capacity(edges_in)
                if not err and parsed is not None:
                    s = nodes_in[0] if nodes_in else None
                    t = nodes_in[-1] if nodes_in else None
                    # Build residual reachability from s
                    flow_dict = resultado.get('flujo_dict') or {}
                    cap_map = {(u, v): int(c) for (u, v, c) in parsed}
                    # residual adjacency: forward residual cap = cap - f ; backward residual cap = f
                    adj = {}
                    for (u, v, c) in parsed:
                        f = int(flow_dict.get(u, {}).get(v, 0))
                        if c - f > 0:
                            adj.setdefault(u, []).append(v)
                        if f > 0:
                            adj.setdefault(v, []).append(u)
                    reachable = set()
                    if s is not None:
                        stack = [s]
                        while stack:
                            u = stack.pop()
                            if u in reachable:
                                continue
                            reachable.add(u)
                            for v in adj.get(u, []):
                                if v not in reachable:
                                    stack.append(v)
                    # min-cut edges from reachable to non-reachable
                    cut = []
                    cut_cap = 0
                    for (u, v, c) in parsed:
                        if u in reachable and v not in reachable:
                            cut.append({"u": u, "v": v, "cap": int(c)})
                            cut_cap += int(c)
                    node_set2 = set(nodes_in)
                    for (uu, vv, _cc) in parsed:
                        node_set2.add(uu)
                        node_set2.add(vv)
                    tset = sorted([n for n in node_set2 if n not in reachable])
                    facts["min_cut"] = {"S": sorted(list(reachable)), "T": tset, "edges": cut, "capacity": cut_cap}
                    # saturated edges (cap==flow)
                    sat = []
                    for (u, v, c) in parsed:
                        f = int(flow_dict.get(u, {}).get(v, 0))
                        if f == int(c) and int(c) > 0:
                            sat.append({"u": u, "v": v, "cap": int(c), "flow": f})
                    facts["saturated_edges"] = sat
                    # certificate
                    fv = resultado.get('flujo_valor')
                    facts["is_optimal"] = (fv is not None and int(fv) == int(cut_cap))
            return facts

        if metodo == 'Flujo de Costo Mínimo':
            facts["flow_value"] = resultado.get('flujo_valor')
            facts["total_cost"] = resultado.get('costo_total')
            facts["flow_dict"] = resultado.get('flujo_dict')
            if datos and edges_in:
                parsed, err = self._parse_directed_capacity_cost(edges_in)
                if not err and parsed is not None:
                    flow_dict = resultado.get('flujo_dict') or {}
                    per_edge = []
                    sat = []
                    for (u, v, cap, cost) in parsed:
                        f = int(flow_dict.get(u, {}).get(v, 0))
                        per_edge.append({"u": u, "v": v, "cap": int(cap), "cost": int(cost), "flow": f, "residual": int(cap) - f})
                        if f == int(cap) and int(cap) > 0:
                            sat.append({"u": u, "v": v, "cap": int(cap), "cost": int(cost), "flow": f})
                    facts["edge_flow_table"] = per_edge
                    facts["saturated_edges"] = sat
            return facts

        # Default
        facts["resultado"] = resultado
        return facts

    def analizar_sensibilidad(self, resultado: Dict[str, Any], datos: Optional[Dict[str, Any]] = None) -> str:
        """Análisis de sensibilidad con guardrails (IA solo interpreta hechos)."""
        try:
            facts = self.construir_contexto_sensibilidad(resultado, datos)
            if 'error' in facts:
                return str(facts['error'])
            # Guardar contexto para la pestaña global de "Sensibilidad"
            set_last_facts("redes", facts)
            return generate_sensitivity_report('redes', facts, api_key=self.api_key, max_retries=1)
        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"

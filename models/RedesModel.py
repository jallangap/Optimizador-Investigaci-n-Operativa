import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai  # Importar la biblioteca de Google Gemini

class RedesModel:
    def __init__(self):
        # Configura tu API Key de Google Gemini
        genai.configure(api_key="AIzaSyBR6xJ3VQWJVK8k6izZyUqbn_gTx9Gvgpk")

    def crear_grafico(self, G):
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black")
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig("red.png")
        plt.close()

    def ruta_mas_corta(self, datos):
        G = nx.Graph()
        for arista in datos['aristas']:
            partes = arista.split('-')
            if len(partes) == 3:
                u, v, w = partes
                G.add_edge(u, v, weight=int(w))
            else:
                return {"error": f"Formato de arista incorrecto: {arista}"}

        self.crear_grafico(G)

        try:
            ruta = nx.shortest_path(G, source=datos['nodos'][0], target=datos['nodos'][-1], weight='weight')
            distancia = nx.shortest_path_length(G, source=datos['nodos'][0], target=datos['nodos'][-1], weight='weight')
            return {"metodo": "Ruta Más Corta", "ruta": ruta, "distancia": distancia}
        except nx.NetworkXNoPath:
            return {"error": "No hay ruta entre los nodos especificados."}

    def arbol_minima_expansion(self, datos):
        G = nx.Graph()
        for arista in datos['aristas']:
            partes = arista.split('-')
            if len(partes) == 3:
                u, v, w = partes
                G.add_edge(u, v, weight=int(w))
            else:
                return {"error": f"Formato de arista incorrecto: {arista}"}

        self.crear_grafico(G)

        try:
            arbol = nx.minimum_spanning_tree(G)
            costo_total = sum(data['weight'] for u, v, data in arbol.edges(data=True))
            return {"metodo": "Árbol de Mínima Expansión", "arbol": list(arbol.edges()), "costo_total": costo_total}
        except nx.NetworkXError:
            return {"error": "No se puede construir un árbol de expansión con los datos proporcionados."}

    def flujo_maximo(self, datos):
        G = nx.DiGraph()
        for arista in datos['aristas']:
            partes = arista.split('-')
            if len(partes) == 3:
                u, v, w = partes
                G.add_edge(u, v, capacity=int(w))
            else:
                return {"error": f"Formato de arista incorrecto: {arista}"}

        self.crear_grafico(G)

        try:
            flujo_valor, flujo_dict = nx.maximum_flow(G, datos['nodos'][0], datos['nodos'][-1])
            return {"metodo": "Flujo Máximo", "flujo_valor": flujo_valor, "flujo_dict": flujo_dict}
        except nx.NetworkXError:
            return {"error": "No se puede calcular el flujo máximo con los datos proporcionados."}

    def analizar_sensibilidad(self, resultado):
        try:
            # Crear un prompt para enviar a la API de Google Gemini
            prompt = (
                f"Realiza un análisis de sensibilidad para el siguiente problema de redes:\n"
                f"Método utilizado: {resultado['metodo']}\n"
                f"Resultado:\n{resultado}\n\n"
                f"Genera un análisis de sensibilidad detallado, incluyendo:\n"
                f"- Cómo afectan los cambios en los pesos de las aristas a la solución.\n"
                f"- Cómo afectan los cambios en la topología de la red a la solución.\n"
                f"- Recomendaciones para optimizar la red."
            )

            # Configurar el modelo de Google Gemini
            model = genai.GenerativeModel('gemini-pro')  # Usar el modelo Gemini Pro

            # Enviar el prompt a la API de Google Gemini
            response = model.generate_content(prompt)

            # Obtener la respuesta generada por la IA
            analisis_sensibilidad = response.text.strip()

            return analisis_sensibilidad
        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"
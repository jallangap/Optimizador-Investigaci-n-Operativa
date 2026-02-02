from models.RedesModel import RedesModel

class RedesController:
    def __init__(self, view):
        """Constructor de la clase RedesController."""
        self.view = view
        self.model = RedesModel()

    def resolver_problema(self, datos):
        """
        Resuelve un problema de redes y RETORNA el resultado.
        La vista espera recibir el diccionario de respuesta.
        """
        metodo = datos.get('metodo')

        if metodo == "Ruta Más Corta":
            resultado = self.model.ruta_mas_corta(datos)
        elif metodo == "Árbol de Mínima Expansión":
            resultado = self.model.arbol_minima_expansion(datos)
        elif metodo == "Flujo Máximo":
            resultado = self.model.flujo_maximo(datos)
        elif metodo == "Flujo de Costo Mínimo":
            resultado = self.model.flujo_costo_minimo(datos)
        else:
            resultado = {"error": "Método no válido."}

        return resultado

    def analizar_sensibilidad(self, resultado, datos_entrada):
        """
        Realiza análisis de sensibilidad pasando el contexto a la IA.
        
        :param resultado: El diccionario con la solución matemática (rutas, flujos, etc.).
        :param datos_entrada: Diccionario con nodos, aristas y el 'contexto' del negocio.
        """
        if resultado is None or isinstance(resultado, str) or "error" in resultado:
            return "Error: No hay un resultado válido para analizar."

        # Delegar al modelo, que ahora sabe manejar el contexto
        return self.model.analizar_sensibilidad(resultado, datos_entrada)
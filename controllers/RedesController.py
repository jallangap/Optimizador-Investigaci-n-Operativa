from models.RedesModel import RedesModel  # Importamos la clase RedesModel desde el módulo models.RedesModel


class RedesController:
    def __init__(self, view):
        """Constructor de la clase RedesController.

        :param view: Referencia a la vista para mostrar resultados.
        """
        self.view = view
        self.model = RedesModel()
        self.resultado_problema = None
        self.ultimo_datos = None

    def resolver_problema(self, datos):
        """Resuelve un problema de redes según el método especificado en 'datos'."""
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
            resultado = "Error: Método no válido."

        self.ultimo_datos = datos
        self.resultado_problema = resultado
        self.view.mostrar_resultado(resultado)

    def analizar_sensibilidad(self):
        """Realiza análisis de sensibilidad sobre la solución obtenida."""
        if self.resultado_problema is None or isinstance(self.resultado_problema, str):
            return "Error: Primero resuelve el problema antes de realizar el análisis de sensibilidad."

        return self.model.analizar_sensibilidad(self.resultado_problema, self.ultimo_datos)

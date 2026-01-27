from models.RedesModel import RedesModel  # Importamos la clase RedesModel desde el módulo models.RedesModel

class RedesController:
    def __init__(self, view):
        """
        Constructor de la clase RedesController.

        :param view: Referencia a la vista para mostrar resultados.
        """
        self.view = view  # Se almacena la vista en el controlador
        self.model = RedesModel()  # Se instancia un objeto de la clase RedesModel
        self.resultado_problema = None  # Variable para almacenar el resultado del problema resuelto

    def resolver_problema(self, datos):
        """
        Método para resolver un problema de redes según el método especificado en 'datos'.
        
        :param datos: Diccionario que contiene la información del problema, incluyendo el método a utilizar.
        """
        metodo = datos['metodo']  # Se extrae el tipo de método de los datos

        # Se selecciona el método adecuado según la opción proporcionada
        if metodo == "Ruta Más Corta":
            resultado = self.model.ruta_mas_corta(datos)  # Llama al método de ruta más corta
        elif metodo == "Árbol de Mínima Expansión":
            resultado = self.model.arbol_minima_expansion(datos)  # Llama al método de árbol de mínima expansión
        elif metodo == "Flujo Máximo":
            resultado = self.model.flujo_maximo(datos)  # Llama al método de flujo máximo
        else:
            resultado = "Error: Método no válido."  # Devuelve un error si el método no es reconocido

        self.resultado_problema = resultado  # Guarda el resultado obtenido
        self.view.mostrar_resultado(resultado)  # Muestra el resultado en la vista

    def analizar_sensibilidad(self):
        """
        Método para realizar un análisis de sensibilidad sobre la solución obtenida.

        :return: Resultado del análisis de sensibilidad o un mensaje de error si no hay resultado previo.
        """
        # Se verifica si hay un resultado previo válido antes de proceder con el análisis
        if self.resultado_problema is None or isinstance(self.resultado_problema, str):
            return "Error: Primero resuelve el problema antes de realizar el análisis de sensibilidad."

        # Llama al método de análisis de sensibilidad del modelo
        resultado_sensibilidad = self.model.analizar_sensibilidad(self.resultado_problema)
        return resultado_sensibilidad  # Retorna el resultado del análisis

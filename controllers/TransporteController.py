from models.TransporteModel import TransporteModel  # Importamos la clase TransporteModel desde el módulo correspondiente

class TransporteController:
    def __init__(self, view):
        """
        Constructor de la clase TransporteController.

        :param view: Referencia a la vista que se usará para mostrar los resultados.
        """
        self.view = view  # Se almacena la vista en el controlador
        self.model = TransporteModel()  # Se instancia un objeto de la clase TransporteModel
        self.resultado_problema = {}  # 🔹 Se inicializa como un diccionario vacío para almacenar resultados por método
        self.ultimo_datos = None  # 🔹 Se almacena la última entrada de datos utilizada

    def resolver_problema(self, datos):
        """
        Método para resolver un problema de transporte según el método especificado en 'datos'.

        :param datos: Diccionario que contiene la información del problema, incluyendo el método a utilizar.
        """
        metodo = datos['metodo']  # Se extrae el método seleccionado de los datos

        # Se selecciona el método adecuado según la opción proporcionada
        if metodo == "Esquina Noroeste":
            resultado = self.model.esquina_noroeste(datos)  # Llama al método de esquina noroeste
        elif metodo == "Costo Mínimo":
            resultado = self.model.costo_minimo(datos)  # Llama al método de costo mínimo
        elif metodo == "Vogel":
            resultado = self.model.vogel(datos)  # Llama al método de aproximación de Vogel
        else:
            resultado = "Error: Método no válido."  # Devuelve un error si el método no es reconocido

        if isinstance(resultado, str):
            self.view.mostrar_resultado(resultado)  # Si el resultado es un error en forma de string, se muestra en la vista
        else:
            # 🔹 Asegurarse de que el resultado tenga la clave 'metodo'
            if 'metodo' not in resultado:
                resultado['metodo'] = metodo  # 🔹 Se añade la clave 'metodo' si no está presente

            self.resultado_problema[metodo] = resultado  # Se almacena el resultado en el diccionario
            self.ultimo_datos = datos  # 🔹 Se guarda la última entrada de datos utilizada
            self.view.mostrar_resultado(resultado)  # Se muestra el resultado en la vista

    def prueba_optimalidad(self, datos_optimalidad):
        """
        Método para ejecutar la prueba de optimalidad con la solución inicial seleccionada.

        :param datos_optimalidad: Diccionario que contiene los datos necesarios para la prueba.
        :return: Resultado de la prueba de optimalidad o un mensaje de error si falta información.
        """
        solucion_inicial = datos_optimalidad.get('solucion_inicial')  # Se obtiene la solución inicial

        if not solucion_inicial:
            return "Error: No se ha seleccionado una solución inicial para la prueba de optimalidad."

        # 🔹 Verificar si hay datos previos almacenados
        if self.ultimo_datos is None:
            return "Error: No hay datos previos para realizar la prueba de optimalidad."

        # Se llama al método de prueba de optimalidad del modelo, pasando los datos previos y la solución inicial
        return self.model.prueba_optimalidad(self.ultimo_datos, solucion_inicial)

    def analizar_sensibilidad(self):
        """
        Método para realizar un análisis de sensibilidad sobre la última solución obtenida.

        :return: Resultado del análisis de sensibilidad o un mensaje de error si no hay datos previos.
        """
        if not self.resultado_problema or isinstance(self.resultado_problema, str):
            return "Error: Primero resuelve el problema antes de realizar el análisis de sensibilidad."

        # 🔹 Obtener el último método utilizado
        ultimo_metodo = list(self.resultado_problema.keys())[-1]  # 🔹 Se obtiene el último método registrado
        resultado = self.resultado_problema.get(ultimo_metodo)  # Se obtiene el resultado asociado a ese método

        if not resultado or 'metodo' not in resultado:
            return "Error: No se encontró un resultado válido para el análisis de sensibilidad."

        # Se llama al método de análisis de sensibilidad del modelo
        resultado_sensibilidad = self.model.analizar_sensibilidad(resultado)
        return resultado_sensibilidad  # Se retorna el resultado del análisis

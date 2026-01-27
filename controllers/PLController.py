from models.PLModel import PLModel  # Importamos la clase PLModel desde el módulo models.PLModel

class PLController:
    def __init__(self, view):
        self.view = view  # Se almacena la vista que se utilizará en el controlador
        self.model = PLModel()  # Se instancia un objeto de la clase PLModel

    def resolver_problema(self, datos):
        """
        Método para resolver un problema de programación lineal según el objetivo especificado en 'datos'.
        
        :param datos: Diccionario que contiene la información del problema, incluyendo el objetivo.
        :return: Resultado de la optimización según el método correspondiente.
        """
        objetivo = datos['objetivo']  # Se extrae el tipo de objetivo de los datos

        # Se selecciona el método adecuado según el objetivo proporcionado
        if objetivo == "Maximizar":
            resultado = self.model.maximizar(datos)  # Llama al método de maximización
        elif objetivo == "Minimizar":
            resultado = self.model.minimizar(datos)  # Llama al método de minimización
        elif objetivo == "Gran M":
            resultado = self.model.gran_m(datos)  # Llama al método del método de la Gran M
        elif objetivo == "Dos Fases":
            resultado = self.model.dos_fases(datos)  # Llama al método del método de dos fases
        elif objetivo == "Dualidad":
            resultado = self.model.dualidad(datos)  # Llama al método de dualidad
        else:
            resultado = "Error: Objetivo no válido."  # Devuelve un error si el objetivo no es reconocido

        return resultado  # Retorna el resultado del método seleccionado

    def analizar_sensibilidad(self, resultado, datos):
        """
        Método para realizar un análisis de sensibilidad sobre la solución obtenida.
        
        :param resultado: Resultado de la optimización obtenida previamente.
        :param datos: Datos originales del problema.
        :return: Resultado del análisis de sensibilidad.
        """
        return self.model.analizar_sensibilidad(resultado, datos)  # Llama al método de análisis de sensibilidad

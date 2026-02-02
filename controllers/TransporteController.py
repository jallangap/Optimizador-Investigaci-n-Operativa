from models.TransporteModel import TransporteModel

class TransporteController:
    def __init__(self, view):
        """
        Constructor de la clase TransporteController.
        :param view: Referencia a la vista.
        """
        self.view = view
        self.model = TransporteModel()
        
        # Almacena resultados por método (ej: {'Vogel': {...}, 'Costo Mínimo': {...}})
        self.resultado_problema = {} 
        
        # Almacena el resultado de la prueba de optimalidad (MODI) por separado
        self.resultado_optimalidad_cache = None
        
        # Guarda los inputs (oferta, demanda, costos) para usarlos en MODI/Sensibilidad
        self.ultimo_datos = None

    def resolver_problema(self, datos):
        """
        Resuelve el problema inicial (Noroeste, Costo Mínimo, Vogel).
        """
        metodo = datos.get('metodo')

        if metodo == "Esquina Noroeste":
            resultado = self.model.esquina_noroeste(datos)
        elif metodo == "Costo Mínimo":
            resultado = self.model.costo_minimo(datos)
        elif metodo == "Vogel":
            resultado = self.model.vogel(datos)
        else:
            resultado = "Error: Método no válido."

        if isinstance(resultado, str):
            self.view.mostrar_resultado(resultado)
        else:
            # Asegurar consistencia de metadatos
            if 'metodo' not in resultado:
                resultado['metodo'] = metodo
            
            # Guardar estado
            self.resultado_problema[metodo] = resultado
            self.ultimo_datos = datos
            
            # Actualizar vista
            self.view.mostrar_resultado(resultado)

    def prueba_optimalidad(self, datos_optimalidad):
        """
        Ejecuta MODI sobre una solución inicial existente.
        """
        solucion_inicial = datos_optimalidad.get('solucion_inicial')

        if not solucion_inicial:
            return {"error": "No hay solución inicial seleccionada."}

        if self.ultimo_datos is None:
            return {"error": "Faltan datos originales (oferta/demanda/costos)."}

        # Llamar al modelo
        res = self.model.prueba_optimalidad(self.ultimo_datos, solucion_inicial)
        
        # Guardar en caché para que 'Analizar Sensibilidad' pueda usarlo si el usuario quiere
        if "error" not in res:
            self.resultado_optimalidad_cache = res
            
        return res

    def analizar_sensibilidad(self, resultado, contexto=""):
        """
        Realiza el análisis con IA.
        
        :param resultado: El diccionario con la solución matemática (puede ser la inicial o la óptima de MODI).
        :param contexto: El texto narrativo que escribió el usuario en la vista.
        """
        if not resultado or isinstance(resultado, str):
            return "Error: Resultado inválido para análisis."

        # Preparamos los datos completos para el modelo
        # Usamos los últimos datos matemáticos (oferta/demanda) y le pegamos el contexto nuevo
        datos_para_modelo = self.ultimo_datos.copy() if self.ultimo_datos else {}
        datos_para_modelo['contexto'] = contexto

        # Delegar al modelo (que llamará a Gemini)
        return self.model.analizar_sensibilidad(resultado, datos_para_modelo)
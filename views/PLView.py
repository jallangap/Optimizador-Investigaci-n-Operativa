from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QComboBox, QHBoxLayout, QGroupBox
)
from controllers.PLController import PLController  # Importa el controlador de Programación Lineal

class PLView(QWidget):
    """
    Vista para la interfaz gráfica de Programación Lineal.
    Permite ingresar datos, resolver problemas y realizar análisis de sensibilidad.
    """
    def __init__(self):
        """
        Constructor de la vista de Programación Lineal.
        """
        super().__init__()
        self.controller = PLController(self)  # Instancia del controlador de PL
        self.resultado_problema = None  # Almacena el resultado del problema resuelto
        self.initUI()  # Inicializa la interfaz de usuario

    def initUI(self):
        """
        Inicializa la interfaz gráfica con los elementos necesarios.
        """
        layout = QVBoxLayout()

        # Sección de selección de métodos básicos y avanzados
        grupo_metodos = QGroupBox("Métodos")  # Agrupar opciones de métodos
        layout_metodos = QVBoxLayout()

        # Selector de objetivo (Maximizar o Minimizar)
        self.label_objetivo = QLabel("Selecciona el objetivo:")
        self.selector_objetivo = QComboBox()
        self.selector_objetivo.addItem("Maximizar")
        self.selector_objetivo.addItem("Minimizar")
        layout_metodos.addWidget(self.label_objetivo)
        layout_metodos.addWidget(self.selector_objetivo)

        # Selector de métodos avanzados
        self.label_metodo_avanzado = QLabel("Selecciona el método avanzado:")
        self.selector_metodo_avanzado = QComboBox()
        self.selector_metodo_avanzado.addItem("Gran M")
        self.selector_metodo_avanzado.addItem("Dos Fases")
        self.selector_metodo_avanzado.addItem("Dualidad")
        layout_metodos.addWidget(self.label_metodo_avanzado)
        layout_metodos.addWidget(self.selector_metodo_avanzado)

        grupo_metodos.setLayout(layout_metodos)
        layout.addWidget(grupo_metodos)

        # Entrada para el número de variables
        self.label_num_variables = QLabel("Número de variables (ej: 2 para x1, x2):")
        self.input_num_variables = QLineEdit()
        layout.addWidget(self.label_num_variables)
        layout.addWidget(self.input_num_variables)

        # Entrada para la función objetivo
        self.label_funcion = QLabel("Función Objetivo (ej: 40*x1 + 30*x2):")
        self.input_funcion = QLineEdit()
        layout.addWidget(self.label_funcion)
        layout.addWidget(self.input_funcion)

        # Entrada para las restricciones
        self.label_restricciones = QLabel("Restricciones (separadas por coma, ej: 2*x1 + x2 <= 100, x1 + x2 <= 80):")
        self.input_restricciones = QLineEdit()
        layout.addWidget(self.label_restricciones)
        layout.addWidget(self.input_restricciones)

        # Botón para resolver el problema
        self.button = QPushButton("Resolver")
        self.button.clicked.connect(self.resolver_problema)
        layout.addWidget(self.button)

        # Área de texto para mostrar resultados
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        layout.addWidget(self.resultado)

        # Área para análisis de sensibilidad
        self.label_sensibilidad = QLabel("Análisis de Sensibilidad:")
        self.sensibilidad = QTextEdit()
        self.sensibilidad.setReadOnly(True)
        layout.addWidget(self.label_sensibilidad)
        layout.addWidget(self.sensibilidad)

        # Botón para ejecutar análisis de sensibilidad
        self.button_sensibilidad = QPushButton("Analizar Sensibilidad")
        self.button_sensibilidad.clicked.connect(self.analizar_sensibilidad)
        layout.addWidget(self.button_sensibilidad)

        self.setLayout(layout)

    def contar_variables(self, texto):
        """
        Cuenta el número de variables únicas en la función objetivo o restricciones.

        :param texto: Expresión matemática en formato de string.
        :return: Número de variables únicas encontradas.
        """
        variables = set()
        for palabra in texto.split():
            if palabra.startswith("x") and palabra[1:].isdigit():
                variables.add(palabra)
        return len(variables)

    def resolver_problema(self):
        """
        Obtiene los datos de la interfaz, los envía al controlador y muestra el resultado.
        """
        objetivo = self.selector_objetivo.currentText()
        metodo_avanzado = self.selector_metodo_avanzado.currentText()
        funcion_obj = self.input_funcion.text().strip().replace(" ", "")  # Eliminar espacios
        restricciones = [r.strip().replace(" ", "") for r in self.input_restricciones.text().strip().split(',')]

        # Obtener el número de variables
        try:
            num_variables = int(self.input_num_variables.text().strip())
            if num_variables <= 0:
                raise ValueError("El número de variables debe ser mayor que 0.")
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Número de variables no válido: {str(e)}")
            return

        if not funcion_obj or not all(restricciones):
            QMessageBox.warning(self, "Error", "Por favor, ingresa la función objetivo y las restricciones.")
            return

        # Verificar que el número de variables coincide con las variables en la función objetivo y restricciones
        variables_funcion = self.contar_variables(funcion_obj)
        variables_restricciones = max([self.contar_variables(r) for r in restricciones])
        if variables_funcion > num_variables or variables_restricciones > num_variables:
            QMessageBox.warning(self, "Error", "El número de variables no coincide con las variables en la función objetivo o restricciones.")
            return

        # Construcción del diccionario de datos
        datos = {
            'num_variables': num_variables,  
            'funcion_obj': funcion_obj,
            'restricciones': restricciones
        }

        # Selección del método para resolver el problema
        if metodo_avanzado == "Gran M":
            self.resultado_problema = self.controller.model.gran_m(datos, objetivo)
        elif metodo_avanzado == "Dos Fases":
            self.resultado_problema = self.controller.model.dos_fases(datos, objetivo)
        elif metodo_avanzado == "Dualidad":
            self.resultado_problema = self.controller.model.dualidad(datos, objetivo)
        else:
            self.resultado_problema = self.controller.model.resolver_problema(datos, objetivo)

        # Mostrar los resultados
        self.mostrar_resultado(self.resultado_problema)

    def analizar_sensibilidad(self):
        """
        Realiza el análisis de sensibilidad sobre la solución obtenida.
        """
        if self.resultado_problema is None or isinstance(self.resultado_problema, str):
            QMessageBox.warning(self, "Error", "Primero resuelve el problema antes de realizar el análisis de sensibilidad.")
            return

        objetivo = self.selector_objetivo.currentText()
        funcion_obj = self.input_funcion.text().strip()
        restricciones = self.input_restricciones.text().strip().split(',')

        datos = {
            'funcion_obj': funcion_obj,
            'restricciones': restricciones
        }

        resultado_sensibilidad = self.controller.model.analizar_sensibilidad(self.resultado_problema, datos)
        self.sensibilidad.setText(resultado_sensibilidad)

    def mostrar_resultado(self, resultado):
        """
        Muestra el resultado obtenido en la interfaz gráfica.

        :param resultado: Diccionario con los valores de las variables y la función objetivo.
        """
        if isinstance(resultado, str):
            self.resultado.setText(f"Error:\n{resultado}")
        else:
            texto_resultado = "Resultado:\n"
            for var, valor in resultado.items():
                texto_resultado += f"{var}: {valor}\n"
            self.resultado.setText(texto_resultado)

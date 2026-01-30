from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QTabWidget
)
from controllers.TransporteController import TransporteController

class TransporteView(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = TransporteController(self)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Crear un QTabWidget para separar las secciones
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Pestaña para resolver el problema
        self.tab_resolver = QWidget()
        self.tabs.addTab(self.tab_resolver, "Resolver Problema")
        self.init_resolver_tab()

        # Pestaña para el análisis de sensibilidad
        self.tab_sensibilidad = QWidget()
        self.tabs.addTab(self.tab_sensibilidad, "Análisis de Sensibilidad")
        self.init_sensibilidad_tab()

        # Pestaña para la prueba de optimalidad
        self.tab_optimalidad = QWidget()
        self.tabs.addTab(self.tab_optimalidad, "Prueba de Optimalidad")
        self.init_optimalidad_tab()

        self.setLayout(layout)

    def init_resolver_tab(self):
        layout = QVBoxLayout()

        # Entrada para la oferta
        self.label_oferta = QLabel("Oferta (separada por coma, ej: 50, 60, 40):")
        self.input_oferta = QLineEdit()
        layout.addWidget(self.label_oferta)
        layout.addWidget(self.input_oferta)

        # Entrada para la demanda
        self.label_demanda = QLabel("Demanda (separada por coma, ej: 30, 70, 50):")
        self.input_demanda = QLineEdit()
        layout.addWidget(self.label_demanda)
        layout.addWidget(self.input_demanda)

        # Entrada para los costos
        self.label_costos = QLabel("Costos (matriz separada por punto y coma, ej: 2,3,4;5,1,3;3,2,1):")
        self.input_costos = QLineEdit()
        layout.addWidget(self.label_costos)
        layout.addWidget(self.input_costos)

        # Selector de método
        self.label_metodo = QLabel("Selecciona el método:")
        self.selector_metodo = QComboBox()
        self.selector_metodo.addItem("Esquina Noroeste")
        self.selector_metodo.addItem("Costo Mínimo")
        self.selector_metodo.addItem("Vogel")
        layout.addWidget(self.label_metodo)
        layout.addWidget(self.selector_metodo)

        # Botón para resolver
        self.button_resolver = QPushButton("Resolver")
        self.button_resolver.clicked.connect(self.resolver_problema)
        layout.addWidget(self.button_resolver)

        # Área para mostrar resultados
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        layout.addWidget(self.resultado)

        self.tab_resolver.setLayout(layout)

    def init_sensibilidad_tab(self):
        layout = QVBoxLayout()

        # Área para mostrar el análisis de sensibilidad
        self.sensibilidad_resultado = QTextEdit()
        self.sensibilidad_resultado.setReadOnly(True)
        layout.addWidget(self.sensibilidad_resultado)


        # Botón para análisis de sensibilidad
        self.button_sensibilidad = QPushButton("Analizar Sensibilidad")
        self.button_sensibilidad.clicked.connect(self.analizar_sensibilidad)
        layout.addWidget(self.button_sensibilidad)

        self.tab_sensibilidad.setLayout(layout)

    def init_optimalidad_tab(self):
        layout = QVBoxLayout()

        # 🔹 Agregamos la selección de método para la prueba de optimalidad
        self.label_seleccionar_resultado = QLabel("Selecciona el resultado para la prueba de optimalidad:")
        self.selector_resultado = QComboBox()  # 🔹 Ahora está correctamente definido
        self.selector_resultado.addItem("Esquina Noroeste")
        self.selector_resultado.addItem("Costo Mínimo")
        self.selector_resultado.addItem("Vogel")

        layout.addWidget(self.label_seleccionar_resultado)
        layout.addWidget(self.selector_resultado)

        self.button_prueba_optimalidad = QPushButton("Realizar Prueba de Optimalidad")
        self.button_prueba_optimalidad.clicked.connect(
            self.realizar_prueba_optimalidad)  # 🔹 Conectar al método corregido
        layout.addWidget(self.button_prueba_optimalidad)

        self.resultado_optimalidad = QTextEdit()
        self.resultado_optimalidad.setReadOnly(True)
        layout.addWidget(self.resultado_optimalidad)

        self.tab_optimalidad.setLayout(layout)

    def resolver_problema(self):
        # Obtener datos de la interfaz
        oferta = list(map(int, self.input_oferta.text().strip().split(',')))
        demanda = list(map(int, self.input_demanda.text().strip().split(',')))
        costos = [list(map(int, fila.split(','))) for fila in self.input_costos.text().strip().split(';')]
        metodo = self.selector_metodo.currentText()

        # Validar la entrada
        if not oferta or not demanda or not costos:
            QMessageBox.warning(self, "Error", "Por favor, ingresa todos los datos.")
            return

        # Enviar datos al controlador
        datos = {
            'oferta': oferta,
            'demanda': demanda,
            'costos': costos,
            'metodo': metodo
        }
        self.controller.resolver_problema(datos)

    def analizar_sensibilidad(self):
        resultado_sensibilidad = self.controller.analizar_sensibilidad()
        self.sensibilidad_resultado.setText(f"Análisis de Sensibilidad:\n{resultado_sensibilidad}")

    def realizar_prueba_optimalidad(self):
        """
        Obtiene el resultado seleccionado y ejecuta la prueba de optimalidad.
        """
        metodo_seleccionado = self.selector_resultado.currentText()

        # Verificar si el método seleccionado tiene un resultado previo
        resultado = self.controller.resultado_problema.get(metodo_seleccionado, None)

        if resultado is None:
            self.resultado_optimalidad.setText(
                f"Error: Primero debes calcular el método {metodo_seleccionado} antes de realizar la prueba de optimalidad.")
            return

        # Crear diccionario con los datos para la prueba de optimalidad
        datos_optimalidad = {
            'solucion_inicial': resultado
        }

        # Llamar al controlador para ejecutar la prueba de optimalidad
        resultado_optimalidad = self.controller.prueba_optimalidad(datos_optimalidad)

        # Manejar errores si el resultado contiene un mensaje de error
        if "error" in resultado_optimalidad:
            self.resultado_optimalidad.setText(f"Error: {resultado_optimalidad['error']}")
            return

        # Manejo de error si no se encuentra el mensaje
        mensaje = resultado_optimalidad.get("mensaje", "No se generó mensaje.")

        # Mostrar el resultado en la interfaz
        texto_resultado = "Resultado de la Prueba de Optimalidad:\n"
        texto_resultado += f"Mensaje: {mensaje}\n"

        if "costo_total" in resultado_optimalidad:
            texto_resultado += f"Costo Total: {resultado_optimalidad['costo_total']}\n\n"

        texto_resultado += "Asignación Óptima:\n"
        for fila in resultado_optimalidad['asignacion']:
            texto_resultado += f"{fila}\n"

        if "celda_mejorar" in resultado_optimalidad:
            texto_resultado += f"\n🔹 Se recomienda mejorar la celda: {resultado_optimalidad['celda_mejorar']}"

        self.resultado_optimalidad.setText(texto_resultado)

    def mostrar_resultado(self, resultado):
        if isinstance(resultado, str):
            self.resultado.setText(f"Error:\n{resultado}")
        else:
            metodo = resultado.get('metodo', 'N/A')
            asignacion = resultado.get('asignacion', [])
            costo_total = resultado.get('costo_total', 'N/A')

            # Crear una cadena de texto formateada
            resultado_formateado = (
                f"Método utilizado: {metodo}\n\n"
                f"Costo total: {costo_total}\n\n"
                f"Asignación:"
            )

            # Mostrar el resultado formateado en el QTextEdit
            self.resultado.setText(resultado_formateado)

            # Crear una tabla para mostrar la asignación
            self.tabla_asignacion = QTableWidget(self)
            self.tabla_asignacion.setRowCount(len(asignacion))
            self.tabla_asignacion.setColumnCount(len(asignacion[0]))

            # Llenar la tabla con los valores de la asignación
            for i, fila in enumerate(asignacion):
                for j, valor in enumerate(fila):
                    self.tabla_asignacion.setItem(i, j, QTableWidgetItem(str(valor)))

            # Añadir la tabla al layout
            self.tab_resolver.layout().addWidget(self.tabla_asignacion)

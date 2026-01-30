from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QComboBox, \
    QDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from controllers.RedesController import RedesController  # Importa el controlador de redes


class ImagenVentana(QDialog):
    """
    Clase que representa una ventana emergente para mostrar una imagen de la red.
    Se usa para visualizar gráficamente el problema de redes.
    """
    def __init__(self):
        """
        Constructor de la ventana de imagen.
        """
        super().__init__()
        self.setWindowTitle("Gráfico de la Red")  # Título de la ventana
        self.setGeometry(100, 100, 600, 400)  # Dimensiones de la ventana

        layout = QVBoxLayout()

        # Etiqueta para mostrar la imagen
        self.imagen_label = QLabel(self)
        pixmap = QPixmap("red.png")  # Se carga la imagen desde el archivo "red.png"
        self.imagen_label.setPixmap(pixmap)
        self.imagen_label.setScaledContents(True)  # Ajustar imagen al tamaño del QLabel

        layout.addWidget(self.imagen_label)
        self.setLayout(layout)  # Establece el diseño de la ventana


class RedesView(QWidget):
    """
    Vista de la interfaz gráfica para la optimización en redes.
    Permite ingresar nodos, aristas y seleccionar un método de optimización.
    """
    def __init__(self):
        """
        Constructor de la vista de Redes.
        """
        super().__init__()
        self.controller = RedesController(self)  # Instancia del controlador de redes
        self.initUI()  # Inicializa la interfaz de usuario

    def initUI(self):
        """
        Configura la interfaz gráfica con los elementos necesarios.
        """
        layout = QVBoxLayout()

        # Selector de método (primero)
        self.label_metodo = QLabel("Selecciona el método:")
        self.selector_metodo = QComboBox()
        self.selector_metodo.addItem("Ruta Más Corta")
        self.selector_metodo.addItem("Árbol de Mínima Expansión")
        self.selector_metodo.addItem("Flujo Máximo")
        self.selector_metodo.addItem("Flujo de Costo Mínimo")
        layout.addWidget(self.label_metodo)
        layout.addWidget(self.selector_metodo)

        # Entrada de nodos
        self.label_nodos = QLabel("Nodos (separados por coma, ej: A,B,C,D):")
        self.input_nodos = QLineEdit()
        layout.addWidget(self.label_nodos)
        layout.addWidget(self.input_nodos)

        # Entrada de aristas
        self.label_aristas = QLabel("Aristas (separadas por punto y coma, ej: A-B-2,A-C-3,B-D-4):")
        self.input_aristas = QLineEdit()
        layout.addWidget(self.label_aristas)
        layout.addWidget(self.input_aristas)

        # Ajustar hints/formatos según método (requiere que label_aristas exista)
        self.selector_metodo.currentTextChanged.connect(self.actualizar_hints)
        self.actualizar_hints(self.selector_metodo.currentText())

        # Botón para resolver el problema
        self.button_resolver = QPushButton("Resolver")
        self.button_resolver.clicked.connect(self.resolver_problema)
        layout.addWidget(self.button_resolver)

        # Botón para el análisis de sensibilidad
        self.button_sensibilidad = QPushButton("Analizar Sensibilidad")
        self.button_sensibilidad.clicked.connect(self.analizar_sensibilidad)
        layout.addWidget(self.button_sensibilidad)

        # Área para mostrar resultados
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        layout.addWidget(self.resultado)

        # Área para mostrar el análisis de sensibilidad
        self.sensibilidad = QTextEdit()
        self.sensibilidad.setReadOnly(True)
        layout.addWidget(self.sensibilidad)

        self.setLayout(layout)  # Se establece el layout principal

    def resolver_problema(self):
        """
        Obtiene los datos de la interfaz, los envía al controlador y muestra el resultado.
        """
        nodos = self.input_nodos.text().strip().split(',')  # Obtiene los nodos ingresados
        aristas = self.input_aristas.text().strip().split(';')  # Obtiene las aristas ingresadas
        metodo = self.selector_metodo.currentText()  # Obtiene el método seleccionado

        if not nodos or not aristas:
            QMessageBox.warning(self, "Error", "Por favor, ingresa todos los datos.")
            return

        # Se crea el diccionario con los datos para enviar al controlador
        datos = {
            'nodos': nodos,
            'aristas': aristas,
            'metodo': metodo
        }
        self.controller.resolver_problema(datos)  # Se envían los datos al controlador

    def actualizar_hints(self, metodo: str):
        """Actualiza la descripción/formato recomendado de aristas según el método."""
        if metodo == "Flujo de Costo Mínimo":
            self.label_aristas.setText(
                "Aristas (separadas por punto y coma, formato: Origen-Destino-Capacidad-Costo; "
                "ej: A-B-10-2;A-C-5-4;B-D-6-1):"
            )
        elif metodo == "Flujo Máximo":
            self.label_aristas.setText(
                "Aristas (separadas por punto y coma, formato: Origen-Destino-Capacidad; "
                "ej: A-B-10;A-C-5;B-D-6):"
            )
        else:
            self.label_aristas.setText(
                "Aristas (separadas por punto y coma, formato: Nodo1-Nodo2-Peso; "
                "ej: A-B-2;A-C-3;B-D-4):"
            )

    def analizar_sensibilidad(self):
        """
        Llama al controlador para realizar el análisis de sensibilidad y muestra el resultado.
        """
        resultado_sensibilidad = self.controller.analizar_sensibilidad()
        self.sensibilidad.setText(f"Análisis de Sensibilidad:\n{resultado_sensibilidad}")

    def mostrar_resultado(self, resultado):
        """
        Muestra el resultado obtenido en la interfaz gráfica.

        :param resultado: Diccionario con los valores obtenidos de la optimización.
        """
        if isinstance(resultado, dict) and "metodo" in resultado:
            metodo = resultado["metodo"]

            if metodo == "Flujo Máximo":
                flujo_valor = resultado.get("flujo_valor", 0)
                flujo_dict = resultado.get("flujo_dict", {})
                flujo_formateado = "\nFlujo en cada arista:\n"

                for origen, destinos in flujo_dict.items():
                    for destino, flujo in destinos.items():
                        if flujo > 0:
                            flujo_formateado += f"{origen} -> {destino}: {flujo}\n"

                resultado_formateado = (
                    f"Método: {metodo}\n"
                    f"Valor máximo del flujo: {flujo_valor}\n"
                    f"{flujo_formateado}"
                )

            elif metodo == "Flujo de Costo Mínimo":
                flujo_valor = resultado.get("flujo_valor", 0)
                costo_total = resultado.get("costo_total", 0)
                flujo_dict = resultado.get("flujo_dict", {})
                flujo_formateado = "\nFlujo en cada arista:\n"

                for origen, destinos in flujo_dict.items():
                    for destino, flujo in destinos.items():
                        if flujo > 0:
                            flujo_formateado += f"{origen} -> {destino}: {flujo}\n"

                resultado_formateado = (
                    f"Método: {metodo}\n"
                    f"Flujo enviado: {flujo_valor}\n"
                    f"Costo total mínimo: {costo_total}\n"
                    f"{flujo_formateado}"
                )

            elif metodo == "Ruta Más Corta":
                ruta = " -> ".join(resultado.get("ruta", []))
                distancia = resultado.get("distancia", 0)
                resultado_formateado = (
                    f"Método: {metodo}\n"
                    f"Ruta más corta: {ruta}\n"
                    f"Distancia total: {distancia}"
                )

            elif metodo == "Árbol de Mínima Expansión":
                arbol = "\n".join([f"{a[0]} - {a[1]}" for a in resultado.get("arbol", [])])
                costo_total = resultado.get("costo_total", 0)
                resultado_formateado = (
                    f"Método: {metodo}\n"
                    f"Aristas del árbol:\n{arbol}\n"
                    f"Costo total: {costo_total}"
                )
            else:
                resultado_formateado = str(resultado)

            self.resultado.setText(f"Resultado:\n{resultado_formateado}")
        else:
            self.resultado.setText(f"Resultado:\n{resultado}")

        self.mostrar_imagen()

    def mostrar_imagen(self):
        """
        Muestra la imagen del gráfico de la red en una ventana emergente.
        """
        self.ventana_imagen = ImagenVentana()
        self.ventana_imagen.exec_()

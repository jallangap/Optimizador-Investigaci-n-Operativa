from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit

class SensibilidadView(QWidget):
    """
    Clase que representa la interfaz gráfica para el análisis de sensibilidad.
    Permite ingresar datos, ejecutar el análisis y mostrar los resultados.
    """
    def __init__(self):
        """
        Constructor de la vista de Sensibilidad.
        """
        super().__init__()
        self.initUI()  # Inicializa la interfaz de usuario

    def initUI(self):
        """
        Configura la interfaz gráfica con los elementos necesarios.
        """
        layout = QVBoxLayout()

        # Etiqueta de título
        self.label = QLabel("Interfaz de Sensibilidad")

        # Campo de entrada de datos
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("Ingrese los datos del problema")  # Texto de ayuda en la entrada

        # Botón para ejecutar el análisis
        self.button = QPushButton("Analizar")

        # Área de texto para mostrar resultados
        self.resultado = QTextEdit(self)
        self.resultado.setReadOnly(True)  # Hace que el área de texto sea de solo lectura

        # Agregar widgets al layout
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.resultado)

        self.setLayout(layout)  # Establece el diseño de la interfaz

        # Conectar el botón a la función de análisis
        self.button.clicked.connect(self.analizar)

    def analizar(self):
        """
        Obtiene los datos ingresados por el usuario, ejecuta un análisis y muestra el resultado.
        """
        datos = self.input.text()  # Obtiene el texto ingresado
        # Aquí puedes llamar al controlador para analizar el problema
        resultado = f"Resultado para: {datos}"  # Mensaje de ejemplo
        self.resultado.setText(resultado)  # Muestra el resultado en el área de texto

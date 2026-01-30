from PyQt5.QtWidgets import QMainWindow, QTabWidget  # Importación de las clases necesarias de PyQt5
from views.PLView import PLView  # Importa la vista de Programación Lineal
from views.TransporteView import TransporteView  # Importa la vista de Transporte
from views.RedesView import RedesView  # Importa la vista de Redes

class MainWindow(QMainWindow):
    """
    Clase MainWindow que representa la ventana principal de la aplicación.

    Hereda de QMainWindow y contiene un QTabWidget con diferentes pestañas
    para cada tipo de optimización.
    """
    def __init__(self):
        """
        Constructor de la ventana principal. Configura la interfaz gráfica y las pestañas.
        """
        super().__init__()  # Llama al constructor de la clase base QMainWindow
        self.setWindowTitle("Aplicación de Optimización")  # Establece el título de la ventana
        self.setGeometry(100, 100, 800, 600)  # Define la posición y tamaño de la ventana (x, y, ancho, alto)

        # Crear un widget de pestañas (QTabWidget) para organizar las vistas en diferentes secciones
        self.tabs = QTabWidget()

        # Agregar cada vista como una pestaña dentro del QTabWidget
        self.tabs.addTab(PLView(), "Programación Lineal")  # Pestaña para Programación Lineal
        self.tabs.addTab(TransporteView(), "Transporte")  # Pestaña para el Modelo de Transporte
        self.tabs.addTab(RedesView(), "Redes")  # Pestaña para Optimización en Redes
        # Nota: La pestaña "Sensibilidad" (asistente tipo chatbot) es opcional.
        # Se mantiene fuera del flujo principal para evitar confusiones, ya que
        # cada módulo (PL/Transporte/Redes) ya incluye su propio análisis.

        # Establecer el widget de pestañas como el widget central de la ventana
        self.setCentralWidget(self.tabs)

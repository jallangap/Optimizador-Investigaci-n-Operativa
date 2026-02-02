from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, 
    QComboBox, QMessageBox, QDialog, QGroupBox, QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from controllers.RedesController import RedesController
import os

class ImagenVentana(QDialog):
    """Ventana emergente para mostrar el gráfico de la red."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gráfico de la Red")
        self.setGeometry(100, 100, 700, 500)
        layout = QVBoxLayout()
        
        self.imagen_label = QLabel(self)
        self.imagen_label.setAlignment(Qt.AlignCenter)
        
        if os.path.exists("red.png"):
            pixmap = QPixmap("red.png")
            if not pixmap.isNull():
                self.imagen_label.setPixmap(pixmap.scaled(650, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.imagen_label.setText("Error al cargar la imagen.")
        else:
            self.imagen_label.setText("No se generó el archivo de imagen.")
        
        layout.addWidget(self.imagen_label)
        self.setLayout(layout)


class RedesView(QWidget):
    """
    Vista para optimización de Redes con campo de Contexto para IA.
    """
    def __init__(self):
        super().__init__()
        self.controller = RedesController(self)
        self.resultado_actual = None
        self.ultimo_input = None # Almacenar últimos datos para la IA
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        scroll = QScrollArea()
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- 1. CONFIGURACIÓN ---
        grupo_config = QGroupBox("1. Definición de la Red")
        l_config = QVBoxLayout()

        l_config.addWidget(QLabel("Método:"))
        self.selector_metodo = QComboBox()
        self.selector_metodo.addItems([
            "Ruta Más Corta", 
            "Árbol de Mínima Expansión", 
            "Flujo Máximo", 
            "Flujo de Costo Mínimo"
        ])
        self.selector_metodo.currentTextChanged.connect(self.actualizar_hints)
        l_config.addWidget(self.selector_metodo)

        l_config.addWidget(QLabel("Nodos (opcional, ej: A,B,C,D):"))
        self.input_nodos = QLineEdit()
        l_config.addWidget(self.input_nodos)

        self.label_aristas = QLabel("Aristas (separadas por punto y coma):")
        l_config.addWidget(self.label_aristas)
        self.input_aristas = QTextEdit()
        self.input_aristas.setMaximumHeight(80)
        l_config.addWidget(self.input_aristas)
        
        self.actualizar_hints(self.selector_metodo.currentText())

        btn_resolver = QPushButton("Resolver y Graficar")
        btn_resolver.clicked.connect(self.resolver_problema)
        l_config.addWidget(btn_resolver)

        grupo_config.setLayout(l_config)
        layout.addWidget(grupo_config)

        # --- 2. RESULTADOS ---
        grupo_res = QGroupBox("2. Resultados Matemáticos")
        l_res = QVBoxLayout()
        
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        self.resultado.setMaximumHeight(150)
        l_res.addWidget(self.resultado)
        
        grupo_res.setLayout(l_res)
        layout.addWidget(grupo_res)

        # --- 3. ANÁLISIS IA ---
        grupo_ai = QGroupBox("3. Interpretación Gerencial (IA)")
        l_ai = QVBoxLayout()

        l_ai.addWidget(QLabel("Contexto del Negocio (Ej: 'Red de fibra óptica', 'Tuberías de agua'...):"))
        self.input_contexto = QTextEdit()
        self.input_contexto.setPlaceholderText("Describe qué representan los nodos y las aristas...")
        self.input_contexto.setMaximumHeight(60)
        l_ai.addWidget(self.input_contexto)

        btn_ai = QPushButton("Analizar Sensibilidad con IA")
        btn_ai.clicked.connect(self.analizar_sensibilidad)
        l_ai.addWidget(btn_ai)

        self.sensibilidad = QTextEdit()
        self.sensibilidad.setReadOnly(True)
        l_ai.addWidget(self.sensibilidad)

        grupo_ai.setLayout(l_ai)
        layout.addWidget(grupo_ai)

        # Finalizar
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def actualizar_hints(self, metodo: str):
        if metodo == "Flujo de Costo Mínimo":
            self.label_aristas.setText("Aristas (Origen-Destino-Capacidad-Costo; ej: A-B-10-2):")
            self.input_aristas.setPlaceholderText("A-B-10-5; A-C-20-4")
        elif metodo == "Flujo Máximo":
            self.label_aristas.setText("Aristas (Origen-Destino-Capacidad; ej: A-B-10):")
            self.input_aristas.setPlaceholderText("S-A-15; A-T-10")
        else:
            self.label_aristas.setText("Aristas (Nodo1-Nodo2-Peso; ej: A-B-5):")
            self.input_aristas.setPlaceholderText("A-B-4; B-C-10")

    def resolver_problema(self):
        """Captura datos, llama al controlador y ACTUALIZA la vista."""
        nodos_txt = self.input_nodos.text().strip()
        nodos = [n.strip() for n in nodos_txt.split(',')] if nodos_txt else []
        
        aristas_txt = self.input_aristas.toPlainText().strip()
        # Soporta ; o saltos de línea
        aristas = [a.strip() for a in aristas_txt.replace('\n', ';').split(';') if a.strip()]
        
        if not aristas:
            QMessageBox.warning(self, "Error", "Ingresa al menos una arista.")
            return

        metodo = self.selector_metodo.currentText()
        datos = {'nodos': nodos, 'aristas': aristas, 'metodo': metodo}
        
        # Guardar input para la IA
        self.ultimo_input = datos 
        
        try:
            # --- CORRECCIÓN AQUÍ: Capturamos el resultado ---
            self.resultado_actual = self.controller.resolver_problema(datos)
            
            # Y llamamos explícitamente a mostrar el resultado
            self.mostrar_resultado(self.resultado_actual)
        except Exception as e:
            QMessageBox.critical(self, "Error Inesperado", str(e))

    def mostrar_resultado(self, resultado):
        """Muestra texto formateado y abre ventana gráfica."""
        if not resultado:
            return

        if isinstance(resultado, dict) and "error" in resultado:
            self.resultado.setText(f"Error: {resultado['error']}")
            return

        # Formateo de texto
        texto = f"Método: {resultado.get('metodo')}\n"
        
        if "distancia" in resultado:
            texto += f"Distancia Total: {resultado['distancia']}\n"
            texto += f"Ruta: {' -> '.join(resultado['ruta'])}"
        elif "costo_total" in resultado and "arbol" in resultado:
            texto += f"Costo Total: {resultado['costo_total']}\n"
            arbol_str = ", ".join([f"{u}-{v}" for u, v, _ in resultado.get("arbol_detalle", [])])
            texto += f"Aristas: {arbol_str}"
        elif "flujo_valor" in resultado:
            texto += f"Flujo Total: {resultado['flujo_valor']}\n"
            if "costo_total" in resultado:
                texto += f"Costo Mínimo: {resultado['costo_total']}\n"
            
            # Mostrar detalle de flujos > 0
            texto += "\nDetalle de Flujos:\n"
            fd = resultado.get("flujo_dict", {})
            for u, dests in fd.items():
                for v, f in dests.items():
                    if f > 0:
                        texto += f"  {u} -> {v}: {f}\n"

        self.resultado.setText(texto)
        
        # Abrir ventana emergente con la imagen
        self.ventana_imagen = ImagenVentana()
        self.ventana_imagen.exec_()

    def analizar_sensibilidad(self):
        if not self.resultado_actual or "error" in self.resultado_actual:
            QMessageBox.warning(self, "Aviso", "Primero resuelve la red exitosamente.")
            return

        self.sensibilidad.setText("Consultando a Gemini...")
        self.sensibilidad.repaint()

        ctx = self.input_contexto.toPlainText()
        
        # Empaquetar input original + contexto para el controlador
        datos_entrada = self.ultimo_input.copy()
        datos_entrada["contexto"] = ctx

        reporte = self.controller.analizar_sensibilidad(self.resultado_actual, datos_entrada)
        self.sensibilidad.setText(reporte)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, 
    QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView
)
from controllers.TransporteController import TransporteController

class TransporteView(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = TransporteController(self)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Tabs principales
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Resolver
        self.tab_resolver = QWidget()
        self.tabs.addTab(self.tab_resolver, "1. Resolver Problema")
        self.init_resolver_tab()

        # Tab 2: Optimalidad (MODI)
        self.tab_optimalidad = QWidget()
        self.tabs.addTab(self.tab_optimalidad, "2. Prueba de Optimalidad")
        self.init_optimalidad_tab()

        # Tab 3: Sensibilidad (IA)
        self.tab_sensibilidad = QWidget()
        self.tabs.addTab(self.tab_sensibilidad, "3. Análisis con IA")
        self.init_sensibilidad_tab()

        self.setLayout(layout)

    def init_resolver_tab(self):
        layout = QVBoxLayout()

        # Inputs
        layout.addWidget(QLabel("Oferta (separada por coma, ej: 100, 200):"))
        self.input_oferta = QLineEdit()
        layout.addWidget(self.input_oferta)

        layout.addWidget(QLabel("Demanda (separada por coma, ej: 150, 150):"))
        self.input_demanda = QLineEdit()
        layout.addWidget(self.input_demanda)

        layout.addWidget(QLabel("Costos (filas por punto y coma, col por coma):"))
        self.input_costos = QLineEdit()
        self.input_costos.setPlaceholderText("Ej: 8,6,10; 9,12,13")
        layout.addWidget(self.input_costos)

        # Selector
        layout.addWidget(QLabel("Método Inicial:"))
        self.selector_metodo = QComboBox()
        self.selector_metodo.addItems(["Esquina Noroeste", "Costo Mínimo", "Vogel"])
        layout.addWidget(self.selector_metodo)

        # Botón
        btn_resolver = QPushButton("Calcular Solución Inicial")
        btn_resolver.clicked.connect(self.resolver_problema)
        layout.addWidget(btn_resolver)

        # Tabla de Resultados (Matriz)
        self.label_res = QLabel("Matriz de Asignación:")
        layout.addWidget(self.label_res)
        self.tabla_asignacion = QTableWidget()
        layout.addWidget(self.tabla_asignacion)

        # Texto Resumen
        self.resultado_texto = QTextEdit()
        self.resultado_texto.setReadOnly(True)
        self.resultado_texto.setMaximumHeight(60)
        layout.addWidget(self.resultado_texto)

        self.tab_resolver.setLayout(layout)

    def init_optimalidad_tab(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Selecciona el resultado base para optimizar:"))
        self.selector_resultado = QComboBox()
        self.selector_resultado.addItems(["Esquina Noroeste", "Costo Mínimo", "Vogel"])
        layout.addWidget(self.selector_resultado)

        btn_opt = QPushButton("Ejecutar MODI (Optimización)")
        btn_opt.clicked.connect(self.realizar_prueba_optimalidad)
        layout.addWidget(btn_opt)

        self.resultado_optimalidad = QTextEdit()
        self.resultado_optimalidad.setReadOnly(True)
        layout.addWidget(self.resultado_optimalidad)

        self.tab_optimalidad.setLayout(layout)

    def init_sensibilidad_tab(self):
        layout = QVBoxLayout()

        # --- CAMPO DE CONTEXTO (NUEVO) ---
        layout.addWidget(QLabel("Contexto del Negocio (Para la IA):"))
        self.input_contexto = QTextEdit()
        self.input_contexto.setPlaceholderText(
            "Ej: Somos una farmacéutica. O1/O2 son Bodegas en Quito/Guayaquil. "
            "D1/D2 son Hospitales. Queremos minimizar fletes..."
        )
        self.input_contexto.setMaximumHeight(80)
        layout.addWidget(self.input_contexto)
        # ---------------------------------

        btn_ai = QPushButton("Generar Informe Gerencial con IA")
        btn_ai.clicked.connect(self.analizar_sensibilidad)
        layout.addWidget(btn_ai)

        self.sensibilidad_resultado = QTextEdit()
        self.sensibilidad_resultado.setReadOnly(True)
        layout.addWidget(self.sensibilidad_resultado)

        self.tab_sensibilidad.setLayout(layout)

    def resolver_problema(self):
        try:
            oferta = list(map(int, self.input_oferta.text().strip().split(',')))
            demanda = list(map(int, self.input_demanda.text().strip().split(',')))
            raw_costos = self.input_costos.text().strip()
            # Soporta separador ; para filas
            costos = [list(map(int, r.split(','))) for r in raw_costos.split(';')]
            metodo = self.selector_metodo.currentText()

            datos = {'oferta': oferta, 'demanda': demanda, 'costos': costos, 'metodo': metodo}
            self.controller.resolver_problema(datos)
        except ValueError:
            QMessageBox.warning(self, "Error", "Formato de números inválido. Revisa las comas y puntos y coma.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def mostrar_resultado(self, resultado):
        """Renderiza la matriz en la tabla y guarda el estado."""
        if isinstance(resultado, str):
            self.resultado_texto.setText(f"Error: {resultado}")
            return

        asignacion = resultado.get('asignacion', [])
        costo = resultado.get('costo_total', 0)
        
        self.resultado_texto.setText(f"Costo Total: {costo}")

        # Configurar tabla visual
        rows = len(asignacion)
        cols = len(asignacion[0]) if rows > 0 else 0
        self.tabla_asignacion.setRowCount(rows)
        self.tabla_asignacion.setColumnCount(cols)
        
        # Headers O1..Om, D1..Dn
        self.tabla_asignacion.setVerticalHeaderLabels([f"O{i+1}" for i in range(rows)])
        self.tabla_asignacion.setHorizontalHeaderLabels([f"D{j+1}" for j in range(cols)])

        for i in range(rows):
            for j in range(cols):
                val = asignacion[i][j]
                item = QTableWidgetItem(str(val))
                if val > 0:
                    item.setBackground(self.palette().highlight()) # Resaltar asignados
                self.tabla_asignacion.setItem(i, j, item)
        
        # Ajustar tamaño columnas
        header = self.tabla_asignacion.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

    def realizar_prueba_optimalidad(self):
        metodo = self.selector_resultado.currentText()
        # Buscar resultado previo en el caché del controlador
        res_previo = self.controller.resultado_problema.get(metodo)
        
        if not res_previo:
            QMessageBox.warning(self, "Aviso", f"Primero resuelve usando '{metodo}'.")
            return

        datos = {'solucion_inicial': res_previo}
        res = self.controller.prueba_optimalidad(datos)
        
        # Mostrar texto detallado
        if "error" in res:
            self.resultado_optimalidad.setText(f"Error: {res['error']}")
        else:
            txt = (f"Resultado MODI:\n{res.get('mensaje')}\n"
                   f"Nuevo Costo Total: {res.get('costo_total')}\n"
                   f"Matriz Óptima: {res.get('asignacion')}")
            self.resultado_optimalidad.setText(txt)

    def analizar_sensibilidad(self):
        """Envía el resultado actual + Contexto a la IA."""
        # Priorizar el resultado óptimo (MODI) si existe, sino el del método seleccionado
        res_opt = getattr(self.controller, 'resultado_optimalidad_cache', None)
        res_base = self.controller.resultado_problema.get(self.selector_metodo.currentText())
        
        resultado_a_analizar = res_opt if res_opt else res_base

        if not resultado_a_analizar:
            QMessageBox.warning(self, "Aviso", "Resuelve el problema primero.")
            return

        self.sensibilidad_resultado.setText("Analizando con IA...")
        self.sensibilidad_resultado.repaint()

        # Capturar contexto
        ctx = self.input_contexto.toPlainText()
        
        # Llamar controlador
        reporte = self.controller.analizar_sensibilidad(resultado_a_analizar, contexto=ctx)
        self.sensibilidad_resultado.setText(reporte)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, 
    QTextEdit, QMessageBox, QComboBox, QGroupBox, QScrollArea
)
from controllers.PLController import PLController

class PLView(QWidget):
    """
    Vista para la interfaz gráfica de Programación Lineal.
    Permite ingresar datos matemáticos y contexto de negocio para la IA.
    """
    def __init__(self):
        super().__init__()
        self.controller = PLController(self)  # Instancia del controlador
        self.resultado_problema = None        # Almacena el resultado matemático
        self.initUI()

    def initUI(self):
        """Inicializa la interfaz gráfica."""
        # Layout principal
        layout = QVBoxLayout()

        # --- 1. SECCIÓN DE CONFIGURACIÓN ---
        grupo_config = QGroupBox("Configuración del Problema")
        layout_config = QVBoxLayout()

        # Selector de objetivo
        self.label_objetivo = QLabel("Objetivo:")
        self.selector_objetivo = QComboBox()
        self.selector_objetivo.addItems(["Maximizar", "Minimizar"])
        layout_config.addWidget(self.label_objetivo)
        layout_config.addWidget(self.selector_objetivo)

        # Selector de método
        self.label_metodo = QLabel("Método de Resolución:")
        self.selector_metodo_avanzado = QComboBox()
        self.selector_metodo_avanzado.addItems(["Gran M", "Simplex", "Dos Fases", "Dualidad"])
        layout_config.addWidget(self.label_metodo)
        layout_config.addWidget(self.selector_metodo_avanzado)

        # Número de variables
        self.label_num_vars = QLabel("Número de variables (ej: 2):")
        self.input_num_variables = QLineEdit()
        layout_config.addWidget(self.label_num_vars)
        layout_config.addWidget(self.input_num_variables)

        grupo_config.setLayout(layout_config)
        layout.addWidget(grupo_config)

        # --- 2. SECCIÓN DE DATOS MATEMÁTICOS ---
        grupo_datos = QGroupBox("Modelo Matemático")
        layout_datos = QVBoxLayout()

        self.label_funcion = QLabel("Función Objetivo (ej: 50*x1 + 40*x2):")
        self.input_funcion = QLineEdit()
        layout_datos.addWidget(self.label_funcion)
        layout_datos.addWidget(self.input_funcion)

        self.label_restricciones = QLabel("Restricciones (separadas por coma):")
        self.input_restricciones = QLineEdit()
        self.input_restricciones.setPlaceholderText("Ej: 2*x1 + x2 <= 100, x1 + x2 <= 80")
        layout_datos.addWidget(self.label_restricciones)
        layout_datos.addWidget(self.input_restricciones)

        grupo_datos.setLayout(layout_datos)
        layout.addWidget(grupo_datos)

        # --- 3. SECCIÓN DE CONTEXTO (NUEVO PARA LA IA) ---
        grupo_contexto = QGroupBox("Contexto del Negocio (Para la IA)")
        layout_contexto = QVBoxLayout()
        
        self.label_contexto = QLabel("Describe el problema real (Opcional):")
        self.input_contexto = QTextEdit()
        self.input_contexto.setPlaceholderText(
            "Ej: Una fábrica produce Sensores (x1) y Controladores (x2). "
            "R1 es stock de Chips, R2 es horas de Mano de Obra..."
        )
        self.input_contexto.setMaximumHeight(60) # Altura reducida para no ocupar mucho espacio
        layout_contexto.addWidget(self.label_contexto)
        layout_contexto.addWidget(self.input_contexto)
        
        grupo_contexto.setLayout(layout_contexto)
        layout.addWidget(grupo_contexto)

        # --- 4. ACCIONES Y RESULTADOS ---
        self.button_resolver = QPushButton("Resolver Problema")
        self.button_resolver.clicked.connect(self.resolver_problema)
        layout.addWidget(self.button_resolver)

        self.label_res = QLabel("Resultado Matemático:")
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        self.resultado.setMaximumHeight(150)
        layout.addWidget(self.label_res)
        layout.addWidget(self.resultado)

        self.button_sensibilidad = QPushButton("Analizar Sensibilidad con IA")
        self.button_sensibilidad.clicked.connect(self.analizar_sensibilidad)
        layout.addWidget(self.button_sensibilidad)

        self.label_sens = QLabel("Informe de Sensibilidad:")
        self.sensibilidad = QTextEdit()
        self.sensibilidad.setReadOnly(True)
        layout.addWidget(self.label_sens)
        layout.addWidget(self.sensibilidad)

        self.setLayout(layout)

    def contar_variables(self, texto):
        """Cuenta variables únicas tipo x1, x2..."""
        variables = set()
        for palabra in texto.replace("+", " ").replace("-", " ").replace("*", " ").split():
            if palabra.startswith("x") and len(palabra) > 1 and palabra[1:].isdigit():
                variables.add(palabra)
        return len(variables)

    def resolver_problema(self):
        """Captura datos, valida y llama al modelo matemático."""
        objetivo = self.selector_objetivo.currentText()
        metodo = self.selector_metodo_avanzado.currentText()
        
        # Limpieza básica
        funcion_obj = self.input_funcion.text().strip().replace(" ", "")
        raw_restricciones = self.input_restricciones.text().strip()
        
        if not raw_restricciones:
            restricciones = []
        else:
            # Soporta separador por coma o punto y coma
            sep = ";" if ";" in raw_restricciones else ","
            restricciones = [r.strip().replace(" ", "") for r in raw_restricciones.split(sep) if r.strip()]

        # Validación Variables
        try:
            txt_vars = self.input_num_variables.text().strip()
            num_variables = int(txt_vars) if txt_vars else 0
            if num_variables <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Ingresa un número de variables válido (>0).")
            return

        if not funcion_obj or not restricciones:
            QMessageBox.warning(self, "Error", "Faltan la función objetivo o las restricciones.")
            return

        # Construir DTO
        datos = {
            'num_variables': num_variables,
            'funcion_obj': funcion_obj,
            'restricciones': restricciones
        }

        # Ejecutar Lógica (Backend)
        try:
            if metodo == "Simplex":
                self.resultado_problema = self.controller.model.resolver_problema(datos, objetivo)
            elif metodo == "Gran M":
                self.resultado_problema = self.controller.model.gran_m(datos, objetivo)
            elif metodo == "Dos Fases":
                self.resultado_problema = self.controller.model.dos_fases(datos, objetivo)
            elif metodo == "Dualidad":
                self.resultado_problema = self.controller.model.dualidad(datos, objetivo)
            else:
                self.resultado_problema = self.controller.model.resolver_problema(datos, objetivo)
            
            self.mostrar_resultado(self.resultado_problema)
            
            # Limpiar campo de sensibilidad anterior para evitar confusión
            self.sensibilidad.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error de Cálculo", str(e))

    def analizar_sensibilidad(self):
        """Prepara los datos + contexto y llama al análisis de IA."""
        if self.resultado_problema is None or isinstance(self.resultado_problema, str):
            QMessageBox.warning(self, "Aviso", "Primero debes resolver el problema correctamente.")
            return

        # Recuperar datos frescos de la UI
        objetivo = self.selector_objetivo.currentText()
        funcion_obj = self.input_funcion.text().strip()
        raw_restricciones = self.input_restricciones.text().strip()
        sep = ";" if ";" in raw_restricciones else ","
        restricciones = [r.strip() for r in raw_restricciones.split(sep) if r.strip()]

        # --- CAPTURAR CONTEXTO ---
        contexto_usuario = self.input_contexto.toPlainText()

        datos = {
            'num_variables': self.input_num_variables.text().strip(),
            'objetivo': objetivo,
            'funcion_obj': funcion_obj,
            'restricciones': restricciones,
            'contexto': contexto_usuario  # <--- AQUÍ SE ENVÍA EL CONTEXTO A LA IA
        }

        self.sensibilidad.setText("Generando análisis con IA... Por favor espera.")
        self.sensibilidad.repaint() # Forzar actualización visual

        try:
            # Llamada al controlador -> Modelo -> IA
            reporte = self.controller.analizar_sensibilidad(self.resultado_problema, datos)
            self.sensibilidad.setText(reporte)
        except Exception as e:
            self.sensibilidad.setText(f"Error al generar reporte: {str(e)}")

    def mostrar_resultado(self, resultado):
        """Formatea el diccionario de resultados en texto legible."""
        if isinstance(resultado, str):
            self.resultado.setText(f"Aviso del Solver:\n{resultado}")
        else:
            lines = []
            if "Valor Óptimo" in resultado:
                lines.append(f"=== SOLUCIÓN ÓPTIMA: {resultado['Valor Óptimo']} ===")
            
            # Separar variables de decisión de las de holgura
            vars_x = {k: v for k, v in resultado.items() if k.startswith('x')}
            vars_other = {k: v for k, v in resultado.items() if not k.startswith('x') and k != "Valor Óptimo"}
            
            lines.append("\nVariables de Decisión:")
            for k, v in sorted(vars_x.items()):
                lines.append(f"  {k} = {v}")
                
            if vars_other:
                lines.append("\nVariables de Holgura/Exceso/Artific.:")
                for k, v in sorted(vars_other.items()):
                    lines.append(f"  {k} = {v}")

            # Mostrar Duals si existen (para Dualidad)
            if "Duals" in resultado:
                 lines.append(f"\nVariables Duales (Precios Sombra): {resultado['Duals']}")

            self.resultado.setText("\n".join(lines))
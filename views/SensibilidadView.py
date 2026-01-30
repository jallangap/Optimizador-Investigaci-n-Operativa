import os

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QComboBox,
)

from utils.sensitivity_context import get_last_facts, get_last_module
from utils.sensitivity_ai import generate_contextual_answer, generate_theory_answer


class SensibilidadView(QWidget):
    """Pestaña global de Sensibilidad.

    Propósito:
    - Permitir preguntas conceptuales (teoría) sobre Investigación Operativa.
    - O bien, si existe un resultado reciente en PL/Transporte/Redes, responder usando
      el contexto real calculado por el programa (sin inventar cifras).
    """

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Asistente de Sensibilidad (IA)"))

        self.source_selector = QComboBox()
        self.source_selector.addItems(
            [
                "Teoría (sin usar resultados)",
                "Usar último resultado (automático)",
                "Usar PL",
                "Usar Transporte",
                "Usar Redes",
            ]
        )
        layout.addWidget(QLabel("Fuente:"))
        layout.addWidget(self.source_selector)

        layout.addWidget(QLabel("Escribe tu pregunta o caso:"))
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Ejemplo: Basado en precios sombra, ¿conviene aumentar el RHS de una restricción?\n"
            "O bien: Explica qué significa el costo reducido en PL/Transporte."
        )
        layout.addWidget(self.input_text)

        self.button_analizar = QPushButton("Analizar")
        self.button_analizar.clicked.connect(self.analizar)
        layout.addWidget(self.button_analizar)

        layout.addWidget(QLabel("Resultado:"))
        self.resultado = QTextEdit()
        self.resultado.setReadOnly(True)
        layout.addWidget(self.resultado)

        self.setLayout(layout)

    def _pick_context(self):
        """Devuelve (module, facts) o (None, None)."""
        choice = self.source_selector.currentText()
        if choice == "Teoría (sin usar resultados)":
            return None, None

        if choice == "Usar PL":
            return "pl", get_last_facts("pl")
        if choice == "Usar Transporte":
            return "transporte", get_last_facts("transporte")
        if choice == "Usar Redes":
            return "redes", get_last_facts("redes")

        # Automático
        m = get_last_module()
        if not m:
            return None, None
        return m, get_last_facts(m)

    def analizar(self):
        user_text = (self.input_text.toPlainText() or "").strip()
        if not user_text:
            self.resultado.setText("Ingresa una pregunta o un caso para analizar.")
            return

        module, facts = self._pick_context()

        # Si no hay contexto, respondemos teoría.
        if not module or not facts:
            self.resultado.setText(generate_theory_answer(user_text, api_key=self.api_key))
            return

        # Respuesta anclada a hechos del programa.
        try:
            res = generate_contextual_answer(module, facts, user_text, api_key=self.api_key, max_retries=1)
        except Exception as e:
            res = f"Error generando el análisis: {str(e)}"
        self.resultado.setText(res)

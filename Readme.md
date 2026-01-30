# Aplicación de Optimización (Investigación Operativa)

Aplicación de escritorio en **Python + PyQt5** para resolver (en Python puro) problemas clásicos de Investigación Operativa:

## Módulos incluidos

### 1) Programación Lineal
- **Simplex (tableau)**
- **Gran M**
- **Dos Fases**
- **Dualidad (formulación y solución)**

Incluye reporte de resultados (variables, holguras) y un **reporte de sensibilidad** basado en datos calculados (precios sombra / costos reducidos cuando aplica), sin inventar valores.

### 2) Transporte
- Solución inicial: **Esquina Noroeste**, **Costo Mínimo**, **Vogel**
- Prueba/mejora: **MODI (u, v, costos reducidos)**

### 3) Redes
- **Ruta más corta**
- **Árbol de expansión mínima**
- **Flujo máximo**
- **Flujo de costo mínimo**

## Restricción importante (cumplimiento)
El motor matemático está implementado en **Python puro**, sin usar librerías externas de optimización (por ejemplo: PuLP, NetworkX, SciPy, etc.).

## Instalación y ejecución

### Requisitos
- Python **3.10+** recomendado

### Pasos (Windows / Linux / macOS)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

En Windows también puedes usar:
```bash
play.bat
```

### Avisos de VS Code (no son errores)
Si VS Code te sugiere seleccionar el entorno o crear uno, es normal cuando detecta un **venv**. Puedes aceptar la sugerencia (“Yes” / “Create”) o simplemente ignorarla si ya trabajas con el venv.

## IA opcional (Gemini)
Algunos reportes pueden usar IA para redactar una **sección adicional de impacto en la toma de decisiones** (sin cambiar los resultados numéricos). Para que el proyecto funcione en cualquier PC **sin configuración extra**, si no hay API Key se usa un modo **offline** (explicación determinística/teórica sin llamadas externas).

Si quieres activar Gemini:
1. Copia `config.example.json` a `config.json`
2. Pega tu `GEMINI_API_KEY` en `config.json`

> **Nota:** `config.json` está ignorado por Git para que no subas tu clave.

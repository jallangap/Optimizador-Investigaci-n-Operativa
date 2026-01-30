# 🚀 Aplicación de Optimización en Python (PyQt5 + PuLP)

Esta es una aplicación de optimización basada en PyQt5 que permite resolver problemas de:
- Programación Lineal (Método Simplex, Gran M, Dos Fases, Dualidad).
- Modelo de Transporte (Esquina Noroeste, Costo Mínimo, Método de Vogel).
- Optimización en Redes (Ruta más corta, Árbol de mínima expansión, Flujo máximo).
- Análisis de Sensibilidad con integración de Google Gemini.

🗝️ **API Key (Gemini)**

La integración con Gemini usa la variable de entorno `GEMINI_API_KEY`.

- Windows (PowerShell):
  - `$env:GEMINI_API_KEY="TU_API_KEY"`
- Windows (CMD):
  - `set GEMINI_API_KEY=TU_API_KEY`

Si no la defines, el proyecto usa un fallback (para compatibilidad con la versión antigua).

------------------------------------------------------------

📌 Requisitos Previos
Antes de ejecutar la aplicación, asegúrate de tener Python 3.8 o superior instalado.

Para verificar la versión de Python en tu sistema, abre una terminal y ejecuta:
python --version

------------------------------------------------------------

🛠️ Ejecución manual
Para ejecutarlo:

1) Abre una terminal en la carpeta del proyecto
2) Ejecuta:

cd ruta/del/proyecto
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py

------------------------------------------------------------

⚡ Ejecución automática

1) Clona el repositorio o descárgalo.
2) Doble click en play.bat.

La primera vez creará el entorno virtual e instalará dependencias.
Las siguientes veces solo abrirá el programa.

------------------------------------------------------------

🔄 Forzar reinstalación
Si algo falla o actualizaste dependencias:

play.bat --force

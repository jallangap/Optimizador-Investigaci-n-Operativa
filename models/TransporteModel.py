import numpy as np
import google.generativeai as genai

class TransporteModel:
    def __init__(self):
        # Configurar API Key de Google Gemini
        genai.configure(api_key="AIzaSyBR6xJ3VQWJVK8k6izZyUqbn_gTx9Gvgpk")

    def esquina_noroeste(self, datos):
        oferta = datos['oferta'].copy()
        demanda = datos['demanda'].copy()
        costos = np.array(datos['costos'])
        m, n = costos.shape
        asignacion = np.zeros((m, n))
        i, j = 0, 0

        while i < m and j < n:
            cantidad = min(oferta[i], demanda[j])
            asignacion[i][j] = cantidad
            oferta[i] -= cantidad
            demanda[j] -= cantidad

            if oferta[i] == 0:
                i += 1
            if demanda[j] == 0:
                j += 1

        costo_total = np.sum(asignacion * costos)
        return {"metodo": "Esquina Noroeste", "asignacion": asignacion.tolist(), "costo_total": costo_total}

    def costo_minimo(self, datos):
        oferta = datos['oferta'].copy()
        demanda = datos['demanda'].copy()
        costos = np.array(datos['costos'])
        m, n = costos.shape
        asignacion = np.zeros((m, n))

        while True:
            # Encontrar la celda con el costo mínimo
            min_cost = np.inf
            min_i, min_j = -1, -1

            for i in range(m):
                for j in range(n):
                    if oferta[i] > 0 and demanda[j] > 0 and costos[i][j] < min_cost:
                        min_cost = costos[i][j]
                        min_i, min_j = i, j

            if min_i == -1 or min_j == -1:
                break

            cantidad = min(oferta[min_i], demanda[min_j])
            asignacion[min_i][min_j] = cantidad
            oferta[min_i] -= cantidad
            demanda[min_j] -= cantidad

        costo_total = np.sum(asignacion * costos)
        return {"metodo": "Costo Mínimo", "asignacion": asignacion.tolist(), "costo_total": costo_total}

    def vogel(self, datos):
        # 🔹 Convertir oferta y demanda a numpy arrays para evitar errores
        oferta = np.array(datos['oferta'], dtype=np.int64)
        demanda = np.array(datos['demanda'], dtype=np.int64)
        costos = np.array(datos['costos'], dtype=np.int64)

        m, n = costos.shape
        asignacion = np.zeros((m, n), dtype=np.int64)

        while np.any(oferta > 0) and np.any(demanda > 0):  # ✅ Ya no habrá error

            # 🔹 Calcular penalizaciones por fila
            penalizaciones_filas = []
            for i in range(m):
                fila_valida = [costos[i][j] for j in range(n) if demanda[j] > 0]
                if len(fila_valida) > 1:
                    fila_valida.sort()
                    penalizaciones_filas.append(fila_valida[1] - fila_valida[0])
                else:
                    penalizaciones_filas.append(0)

            # 🔹 Calcular penalizaciones por columna
            penalizaciones_columnas = []
            for j in range(n):
                columna_valida = [costos[i][j] for i in range(m) if oferta[i] > 0]
                if len(columna_valida) > 1:
                    columna_valida.sort()
                    penalizaciones_columnas.append(columna_valida[1] - columna_valida[0])
                else:
                    penalizaciones_columnas.append(0)

            # 🔹 Determinar la penalización máxima
            max_pen_fila = max(penalizaciones_filas)
            max_pen_col = max(penalizaciones_columnas)

            if max_pen_fila >= max_pen_col:
                i = penalizaciones_filas.index(max_pen_fila)
                fila_valida = [costos[i][j] for j in range(n) if demanda[j] > 0]
                j = [j for j in range(n) if demanda[j] > 0][fila_valida.index(min(fila_valida))]
            else:
                j = penalizaciones_columnas.index(max_pen_col)
                columna_valida = [costos[i][j] for i in range(m) if oferta[i] > 0]
                i = [i for i in range(m) if oferta[i] > 0][columna_valida.index(min(columna_valida))]

            # 🔹 Asignar la cantidad mínima entre la oferta y la demanda
            cantidad = min(oferta[i], demanda[j])
            asignacion[i][j] = cantidad
            oferta[i] -= cantidad
            demanda[j] -= cantidad

        costo_total = np.sum(asignacion * costos)
        return {"metodo": "Vogel", "asignacion": asignacion.tolist(), "costo_total": int(costo_total)}

    def prueba_optimalidad(self, datos, solucion_inicial):
        """
        Implementación del método MODI para la prueba de optimalidad.
        """
        if solucion_inicial is None:
            return {"error": "No se proporcionó una solución inicial."}

        oferta = datos['oferta']
        demanda = datos['demanda']
        costos = np.array(datos['costos'])
        asignacion = np.array(solucion_inicial['asignacion'])

        m, n = costos.shape
        u = [None] * m  # Potenciales de filas
        v = [None] * n  # Potenciales de columnas
        u[0] = 0  # Fijamos el primer valor arbitrariamente

        # Paso 1: Resolver u y v usando las asignaciones
        for _ in range(m + n):
            for i in range(m):
                for j in range(n):
                    if asignacion[i][j] > 0:  # Solo consideramos celdas asignadas
                        if u[i] is not None and v[j] is None:
                            v[j] = costos[i][j] - u[i]
                        elif u[i] is None and v[j] is not None:
                            u[i] = costos[i][j] - v[j]

        # Si no se pudieron calcular todos los valores de u y v, la solución es degenerada
        if None in u or None in v:
            return {
                "error": "La solución es degenerada, no se pueden calcular todos los u y v.",
                "mensaje": "No se puede evaluar la optimalidad debido a datos insuficientes."
            }

        # Paso 2: Calcular costos reducidos
        costos_reducidos = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if asignacion[i][j] == 0:  # Solo consideramos celdas NO asignadas
                    costos_reducidos[i][j] = costos[i][j] - (u[i] + v[j])

        # Paso 3: Verificar si la solución es óptima
        costo_total = np.sum(asignacion * costos)  # 🔹 Se asegura de calcular siempre el costo total

        if np.all(costos_reducidos >= 0):
            return {
                "metodo": "Prueba de Optimalidad",
                "mensaje": "La solución es óptima.",
                "asignacion": asignacion.tolist(),
                "costo_total": costo_total,
                "u": u,
                "v": v,
                "costos_reducidos": costos_reducidos.tolist()
            }

        # Paso 4: Si la solución no es óptima, mejorarla
        i_min, j_min = np.unravel_index(np.argmin(costos_reducidos), costos_reducidos.shape)

        return {
            "metodo": "Prueba de Optimalidad",
            "mensaje": "La solución NO es óptima. Se recomienda mejorar la asignación.",
            "asignacion": asignacion.tolist(),
            "costo_total": costo_total,
            "u": u,
            "v": v,
            "costos_reducidos": costos_reducidos.tolist(),
            "celda_mejorar": (i_min, j_min)
        }

    def analizar_sensibilidad(self, resultado):
        try:
            # Crear un prompt para enviar a la API de Google Gemini
            prompt = (
                f"Realiza un análisis de sensibilidad para el siguiente problema de transporte:\n"
                f"Método utilizado: {resultado['metodo']}\n"
                f"Asignación:\n{resultado['asignacion']}\n"
                f"Costo total: {resultado['costo_total']}\n\n"
                f"Genera un análisis de sensibilidad detallado, incluyendo:\n"
                f"- Cómo afectan los cambios en los costos al costo total.\n"
                f"- Cómo afectan los cambios en la oferta y la demanda a la asignación.\n"
                f"- Recomendaciones para optimizar el costo total."
            )

            # Configurar el modelo de Google Gemini
            model = genai.GenerativeModel('gemini-pro')  # Usar el modelo Gemini Pro

            # Enviar el prompt a la API de Google Gemini
            response = model.generate_content(prompt)

            # Obtener la respuesta generada por la IA
            analisis_sensibilidad = response.text.strip()

            return analisis_sensibilidad
        except Exception as e:
            return f"Error en el análisis de sensibilidad: {str(e)}"
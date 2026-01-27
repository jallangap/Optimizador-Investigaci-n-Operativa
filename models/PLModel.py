from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, value

import google.generativeai as genai

class PLModel:
    def __init__(self):
        # Configura tu API Key de Google Gemini
        genai.configure(api_key="AIzaSyBR6xJ3VQWJVK8k6izZyUqbn_gTx9Gvgpk")  # Reemplaza con tu API Key

    def resolver_problema(self, datos, objetivo="Maximizar"):
        """
        Resuelve el problema de PL simple agregando variables de holgura (para <=)
        y de excedente (para >=) en cada restricción.
        """
        if objetivo == "Maximizar":
            problema = LpProblem("Maximizar", LpMaximize)
        else:
            problema = LpProblem("Minimizar", LpMinimize)

        num_variables = datos['num_variables']
        # Crear variables de decisión
        x = {f"x{i+1}": LpVariable(f"x{i+1}", lowBound=0) for i in range(num_variables)}

        try:
            # Función objetivo
            funcion_obj = eval(datos['funcion_obj'], {}, x)
            problema += funcion_obj, "Objetivo"

            # Diccionario para almacenar variables de holgura/excedente
            extras = {}

            # Procesar cada restricción
            for idx, restriccion in enumerate(datos['restricciones']):
                # Reemplazar símbolos especiales
                restriccion = restriccion.replace("≤", "<=").replace("≥", ">=")
                if "<=" in restriccion:
                    lhs, rhs = restriccion.split("<=")
                    s = LpVariable(f"s{idx+1}", lowBound=0)
                    extras[f"s{idx+1}"] = s
                    # Transformar: LHS + s = RHS
                    problema += eval(lhs.strip(), {}, x) + s == eval(rhs.strip(), {}, {})
                elif ">=" in restriccion:
                    lhs, rhs = restriccion.split(">=")
                    e = LpVariable(f"e{idx+1}", lowBound=0)
                    extras[f"e{idx+1}"] = e
                    # Transformar: LHS - e = RHS
                    problema += eval(lhs.strip(), {}, x) - e == eval(rhs.strip(), {}, {})
                elif "=" in restriccion:
                    lhs, rhs = restriccion.split("=")
                    problema += eval(lhs.strip(), {}, x) == eval(rhs.strip(), {}, {})
                else:
                    raise ValueError(f"Formato de restricción inválido: {restriccion}")

            # Resolver el problema
            problema.solve()

            # Recopilar resultados (variables de decisión y extras)
            resultado = {var: value(x[var]) for var in x}
            for var in extras:
                resultado[var] = value(extras[var])
            resultado["Valor Óptimo"] = value(problema.objective)

            return resultado

        except Exception as e:
            return f"Error al resolver el problema: {str(e)}"

    def gran_m(self, datos, objetivo="Maximizar"):
        """
        Método Gran M: se añaden variables artificiales para restricciones ≥ y =
        y se penalizan en la función objetivo.
        """
        if objetivo == "Maximizar":
            problema = LpProblem("Gran M", LpMaximize)
        else:
            problema = LpProblem("Gran M", LpMinimize)

        num_variables = datos['num_variables']
        x = {f"x{i+1}": LpVariable(f"x{i+1}", lowBound=0) for i in range(num_variables)}
        M = 1e6  # Valor grande
        try:
            # Función objetivo original
            func_obj = eval(datos['funcion_obj'], {}, x)
            # Inicialmente se agrega la función objetivo sin penalización
            problema += func_obj, "Objetivo"

            # Diccionarios para variables extra
            extras = {}      # Para holgura o excedente
            artificiales = {}  # Para variables artificiales

            for idx, restriccion in enumerate(datos['restricciones']):
                restriccion = restriccion.replace("≤", "<=").replace("≥", ">=")
                if "<=" in restriccion:
                    lhs, rhs = restriccion.split("<=")
                    s = LpVariable(f"s{idx+1}", lowBound=0)
                    extras[f"s{idx+1}"] = s
                    problema += eval(lhs.strip(), {}, x) + s == eval(rhs.strip(), {}, {})
                elif ">=" in restriccion:
                    lhs, rhs = restriccion.split(">=")
                    e = LpVariable(f"e{idx+1}", lowBound=0)
                    a = LpVariable(f"a{idx+1}", lowBound=0)
                    extras[f"e{idx+1}"] = e
                    artificiales[f"a{idx+1}"] = a
                    # Restricción transformada: LHS - e + a = RHS
                    problema += eval(lhs.strip(), {}, x) - e + a == eval(rhs.strip(), {}, {})
                elif "=" in restriccion:
                    lhs, rhs = restriccion.split("=")
                    a = LpVariable(f"a{idx+1}", lowBound=0)
                    artificiales[f"a{idx+1}"] = a
                    # Restricción transformada: LHS + a = RHS
                    problema += eval(lhs.strip(), {}, x) + a == eval(rhs.strip(), {}, {})
                else:
                    raise ValueError(f"Formato de restricción inválido: {restriccion}")

            # Penalizar las variables artificiales en la función objetivo
            if artificiales:
                if objetivo == "Maximizar":
                    problema.objective += - M * sum(artificiales.values())
                else:
                    problema.objective += M * sum(artificiales.values())

            problema.solve()

            resultado = {var: value(x[var]) for var in x}
            # Incluir variables de holgura, excedente y artificiales
            for v in problema.variables():
                if v.name.startswith("s") or v.name.startswith("e") or v.name.startswith("a"):
                    resultado[v.name] = value(v)
            resultado["Valor Óptimo"] = value(problema.objective)
            return resultado

        except Exception as e:
            return f"Error al resolver el problema: {str(e)}"

    def dos_fases(self, datos, objetivo="Maximizar"):
        """
        Método de Dos Fases:
         - Fase 1: Se minimiza la suma de las variables artificiales para forzarlas a 0.
         - Fase 2: Se resuelve el problema original (sin las artificiales) si la Fase 1 es exitosa.
        """
        try:
            num_variables = datos['num_variables']
            # Fase 1: Minimización de artificiales
            prob1 = LpProblem("Fase 1", LpMinimize)
            x = {f"x{i+1}": LpVariable(f"x{i+1}", lowBound=0) for i in range(num_variables)}
            artificiales = {}
            slack_vars = {}

            for idx, restriccion in enumerate(datos['restricciones']):
                restriccion = restriccion.replace("≤", "<=").replace("≥", ">=")
                if "<=" in restriccion:
                    lhs, rhs = restriccion.split("<=")
                    s = LpVariable(f"s{idx+1}", lowBound=0)
                    slack_vars[f"s{idx+1}"] = s
                    prob1 += eval(lhs.strip(), {}, x) + s == eval(rhs.strip(), {}, {})
                elif ">=" in restriccion:
                    lhs, rhs = restriccion.split(">=")
                    e = LpVariable(f"e{idx+1}", lowBound=0)
                    a = LpVariable(f"a{idx+1}", lowBound=0)
                    artificiales[f"a{idx+1}"] = a
                    prob1 += eval(lhs.strip(), {}, x) - e + a == eval(rhs.strip(), {}, {})
                elif "=" in restriccion:
                    lhs, rhs = restriccion.split("=")
                    a = LpVariable(f"a{idx+1}", lowBound=0)
                    artificiales[f"a{idx+1}"] = a
                    prob1 += eval(lhs.strip(), {}, x) + a == eval(rhs.strip(), {}, {})
                else:
                    raise ValueError(f"Formato de restricción inválido: {restriccion}")

            # Objetivo de Fase 1: minimizar la suma de las artificiales
            prob1 += sum(artificiales.values()), "Fase1_Objetivo"
            prob1.solve()
            total_artificial = sum([value(a) for a in artificiales.values()])
            if abs(total_artificial) > 1e-5:
                return "Problema infactible: suma de artificiales > 0 en Fase 1"

            # Fase 2: Resolver el problema original sin artificiales
            if objetivo == "Maximizar":
                prob2 = LpProblem("Fase 2", LpMaximize)
            else:
                prob2 = LpProblem("Fase 2", LpMinimize)
            func_obj = eval(datos['funcion_obj'], {}, x)
            prob2 += func_obj, "Objetivo"

            # Reconstruir las restricciones sin las artificiales
            for idx, restriccion in enumerate(datos['restricciones']):
                restriccion = restriccion.replace("≤", "<=").replace("≥", ">=")
                if "<=" in restriccion:
                    lhs, rhs = restriccion.split("<=")
                    s = LpVariable(f"s{idx+1}", lowBound=0)
                    prob2 += eval(lhs.strip(), {}, x) + s == eval(rhs.strip(), {}, {})
                elif ">=" in restriccion:
                    lhs, rhs = restriccion.split(">=")
                    e = LpVariable(f"e{idx+1}", lowBound=0)
                    prob2 += eval(lhs.strip(), {}, x) - e == eval(rhs.strip(), {}, {})
                elif "=" in restriccion:
                    lhs, rhs = restriccion.split("=")
                    prob2 += eval(lhs.strip(), {}, x) == eval(rhs.strip(), {}, {})
            prob2.solve()
            resultado = {var: value(x[var]) for var in x}
            # Incluir las variables de holgura/excedente de Fase 2
            for v in prob2.variables():
                if v.name.startswith("s") or v.name.startswith("e"):
                    resultado[v.name] = value(v)
            resultado["Valor Óptimo"] = value(prob2.objective)
            return resultado

        except Exception as e:
            return f"Error al resolver el problema en Dos Fases: {str(e)}"

    def dualidad(self, datos, objetivo="Maximizar"):
        """
        Método Dualidad:
         - Se arma el problema primal agregando variables de holgura/excedente.
         - Luego se extraen los precios duales (atributo 'pi') de cada restricción.
           Nota: Para obtener los duales es necesario que el solver utilizado (ej. CBC) los soporte.
        """
        if objetivo == "Maximizar":
            problema = LpProblem("Primal", LpMaximize)
        else:
            problema = LpProblem("Primal", LpMinimize)

        num_variables = datos['num_variables']
        x = {f"x{i+1}": LpVariable(f"x{i+1}", lowBound=0) for i in range(num_variables)}

        try:
            func_obj = eval(datos['funcion_obj'], {}, x)
            problema += func_obj, "Objetivo"

            # Lista para guardar restricciones (para luego extraer duales)
            restricciones_list = []
            for idx, restriccion in enumerate(datos['restricciones']):
                restriccion = restriccion.replace("≤", "<=").replace("≥", ">=")
                if "<=" in restriccion:
                    lhs, rhs = restriccion.split("<=")
                    s = LpVariable(f"s{idx+1}", lowBound=0)
                    constr = eval(lhs.strip(), {}, x) + s == eval(rhs.strip(), {}, {})
                    problema += constr, f"Restriccion_{idx+1}"
                    restricciones_list.append(problema.constraints[f"Restriccion_{idx+1}"])
                elif ">=" in restriccion:
                    lhs, rhs = restriccion.split(">=")
                    e = LpVariable(f"e{idx+1}", lowBound=0)
                    constr = eval(lhs.strip(), {}, x) - e == eval(rhs.strip(), {}, {})
                    problema += constr, f"Restriccion_{idx+1}"
                    restricciones_list.append(problema.constraints[f"Restriccion_{idx+1}"])
                elif "=" in restriccion:
                    lhs, rhs = restriccion.split("=")
                    constr = eval(lhs.strip(), {}, x) == eval(rhs.strip(), {}, {})
                    problema += constr, f"Restriccion_{idx+1}"
                    restricciones_list.append(problema.constraints[f"Restriccion_{idx+1}"])
                else:
                    raise ValueError(f"Formato de restricción inválido: {restriccion}")

            problema.solve()

            resultado = {var: value(x[var]) for var in x}
            duales = {}
            # Extraer los precios duales de cada restricción (si el solver los provee)
            for i, constr in enumerate(restricciones_list):
                duales[f"Dual_{i+1}"] = constr.pi  if hasattr(constr, "pi") else None
            resultado["Duals"] = duales
            resultado["Valor Óptimo"] = value(problema.objective)
            return resultado

        except Exception as e:
            return f"Error al resolver el problema en Dualidad: {str(e)}"

    def analizar_sensibilidad(self, resultado, datos):
        try:
            # Crear un prompt para enviar a la API de Google Gemini
            prompt = (
                f"Realiza un análisis de sensibilidad para el siguiente problema de programación lineal:\n"
                f"Función objetivo: {datos['funcion_obj']}\n"
                f"Restricciones: {', '.join(datos['restricciones'])}\n"
                f"Resultados:\n"
                f"{', '.join([f'{var}: {valor}' for var, valor in resultado.items()])}\n\n"
                f"Genera un análisis de sensibilidad detallado, incluyendo:\n"
                f"- OBJECTIVE FUNCTION VALUE\n"
                f"- VARIABLE VALUE y REDUCED COST\n"
                f"- ROW SLACK OR SURPLUS y DUAL PRICES\n"
                f"- RANGES IN WHICH THE BASIS IS UNCHANGED\n"
                f"- Predicción de la función objetivo si las variables aumentan en 1."
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
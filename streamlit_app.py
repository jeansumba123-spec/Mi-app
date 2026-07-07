# ============================================================
# APP STREAMLIT - PROGRAMACIÓN NO LINEAL
# Versión corregida, robusta y lista para informe académico
# Universidad de Cuenca - Investigación de Operaciones
# ============================================================
#
# Requisitos de instalación:
#   pip install streamlit numpy sympy scipy plotly pandas
#
# Para ejecutar:
#   streamlit run app_programacion_no_lineal.py
#
# Nota:
# Este programa está diseñado con fines académicos. Para problemas no convexos,
# ningún método numérico local garantiza por sí solo el óptimo global.
# Por eso se incluyen verificaciones, clasificación Hessiana, restricciones,
# candidatos, KKT aproximado y mensajes de advertencia.
# ============================================================

import itertools
import math
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as opt
import streamlit as st
import sympy as sp


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Programación No Lineal",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        background-color: #0F3050;
        padding: 14px 24px;
        border-radius: 10px;
        color: white;
        margin-bottom: 18px;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 2px;
        font-size: 1.7rem;
    }
    .main-header h3 {
        color: #A0C0DF;
        margin-top: 0px;
        margin-bottom: 5px;
        font-size: 1.05rem;
        font-weight: normal;
    }
    .main-header p {
        color: #D0E0EF;
        font-size: 0.9rem;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #F6F8FA;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E6E8EB;
        margin-bottom: 8px;
    }
    .stButton>button {
        background-color: #008C7A;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        width: 100%;
        margin-top: 12px;
    }
    .stButton>button:hover {
        background-color: #006B5D;
        color: white;
    }
    </style>
    <div class="main-header">
        <h1>PROGRAMACIÓN NO LINEAL</h1>
        <h3>Aplicación académica corregida y robusta</h3>
        <p><b>Materia:</b> Investigación de Operaciones | <b>Área:</b> Optimización no lineal</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES DE VALIDACIÓN Y PARSEO
# ============================================================

def mostrar_error_amigable(titulo: str, exc: Exception) -> None:
    """Muestra errores de forma entendible para el usuario."""
    st.error(titulo)
    with st.expander("Ver detalle técnico del error"):
        st.code("".join(traceback.format_exception_only(type(exc), exc)))


def variables_por_n(n: int):
    """Devuelve símbolos x1, x2, ..., xn."""
    return sp.symbols(" ".join([f"x{i+1}" for i in range(n)]))


def validar_dimension_vector(x: np.ndarray, n: int, nombre: str = "vector") -> None:
    if len(x) != n:
        raise ValueError(f"El {nombre} debe tener {n} valores, pero se ingresaron {len(x)}.")


def validar_dimension_matriz(M: np.ndarray, filas: int, columnas: int, nombre: str = "matriz") -> None:
    if M.shape != (filas, columnas):
        raise ValueError(
            f"La {nombre} debe ser de dimensión {filas}x{columnas}, "
            f"pero se ingresó {M.shape[0]}x{M.shape[1]}."
        )


def parse_vector(texto: str) -> np.ndarray:
    """
    Convierte un texto tipo:
        1, 2, 3
    en np.array([1, 2, 3]).
    """
    try:
        valores = [float(x.strip()) for x in texto.replace(";", ",").split(",") if x.strip() != ""]
        if not valores:
            raise ValueError("No se ingresaron valores.")
        return np.array(valores, dtype=float)
    except Exception as exc:
        raise ValueError("El vector debe escribirse con números separados por comas. Ejemplo: 1, 2, 3") from exc


def parse_matrix(texto: str) -> np.ndarray:
    """
    Convierte un texto tipo:
        1, 0
        0, 1
    en una matriz NumPy.
    """
    try:
        filas = []
        for linea in texto.strip().splitlines():
            if linea.strip():
                filas.append([float(x.strip()) for x in linea.replace(";", ",").split(",") if x.strip() != ""])
        if not filas:
            raise ValueError("No se ingresaron filas.")
        longitudes = [len(f) for f in filas]
        if len(set(longitudes)) != 1:
            raise ValueError("Todas las filas deben tener el mismo número de columnas.")
        return np.array(filas, dtype=float)
    except Exception as exc:
        raise ValueError(
            "La matriz debe escribirse por filas, separando columnas con comas. "
            "Ejemplo:\n2, 0\n0, 2"
        ) from exc


def parse_restrictions(texto: str, n: int):
    """
    Convierte restricciones escritas como:
        1,0,4,<=
        0,1,6,<=
        3,2,18,<=

    Cada fila representa:
        a1*x1 + a2*x2 + ... + an*xn <= b

    También acepta >= y =.
    Internamente devuelve:
        A_ineq x <= b_ineq
        A_eq x = b_eq
    """
    A_ineq = []
    b_ineq = []
    A_eq = []
    b_eq = []

    if not texto.strip():
        return np.empty((0, n)), np.empty((0,)), np.empty((0, n)), np.empty((0,))

    for idx, linea in enumerate(texto.strip().splitlines(), start=1):
        if not linea.strip():
            continue

        partes = [p.strip() for p in linea.replace(";", ",").split(",")]
        if len(partes) != n + 2:
            raise ValueError(
                f"La restricción en la línea {idx} debe tener {n} coeficientes, "
                f"un valor independiente y un signo. Ejemplo para {n} variables: "
                + ",".join(["1"] * n)
                + ",10,<="
            )

        coeffs = [float(p) for p in partes[:n]]
        valor = float(partes[n])
        signo = partes[n + 1].replace(" ", "")

        if signo in ["<=", "≤", "=<"]:
            A_ineq.append(coeffs)
            b_ineq.append(valor)
        elif signo in [">=", "≥", "=>"]:
            A_ineq.append([-c for c in coeffs])
            b_ineq.append(-valor)
        elif signo in ["=", "=="]:
            A_eq.append(coeffs)
            b_eq.append(valor)
        else:
            raise ValueError(
                f"Signo inválido en la línea {idx}: '{signo}'. Use <=, >= o =."
            )

    return (
        np.array(A_ineq, dtype=float).reshape(-1, n),
        np.array(b_ineq, dtype=float),
        np.array(A_eq, dtype=float).reshape(-1, n),
        np.array(b_eq, dtype=float),
    )


def restricciones_scipy(A_ineq, b_ineq, A_eq=None, b_eq=None):
    """Crea restricciones compatibles con scipy.optimize.minimize."""
    cons = []

    if A_ineq is not None and len(A_ineq) > 0:
        for a, b in zip(A_ineq, b_ineq):
            cons.append({
                "type": "ineq",
                "fun": lambda x, a=a, b=b: float(b - np.dot(a, x)),
                "jac": lambda x, a=a, b=b: -np.array(a, dtype=float),
            })

    if A_eq is not None and len(A_eq) > 0:
        for a, b in zip(A_eq, b_eq):
            cons.append({
                "type": "eq",
                "fun": lambda x, a=a, b=b: float(np.dot(a, x) - b),
                "jac": lambda x, a=a, b=b: np.array(a, dtype=float),
            })

    return cons


def es_factible(x, A_ineq, b_ineq, A_eq=None, b_eq=None, tol=1e-6):
    """Verifica factibilidad con tolerancia."""
    x = np.array(x, dtype=float)
    ok_ineq = True
    ok_eq = True

    if A_ineq is not None and len(A_ineq) > 0:
        ok_ineq = np.all(A_ineq @ x <= b_ineq + tol)

    if A_eq is not None and len(A_eq) > 0:
        ok_eq = np.all(np.abs(A_eq @ x - b_eq) <= tol)

    return bool(ok_ineq and ok_eq)


def proyectar_punto(x0, A_ineq, b_ineq, A_eq=None, b_eq=None):
    """Proyecta un punto sobre la región factible lineal."""
    x0 = np.array(x0, dtype=float)
    cons = restricciones_scipy(A_ineq, b_ineq, A_eq, b_eq)

    res = opt.minimize(
        fun=lambda x: float(np.sum((x - x0) ** 2)),
        x0=x0,
        jac=lambda x: 2 * (x - x0),
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )

    if not res.success:
        raise ValueError(
            "No se pudo proyectar el punto inicial sobre la región factible. "
            "Revise si las restricciones tienen solución."
        )

    return res.x


# ============================================================
# FUNCIONES SIMBÓLICAS
# ============================================================

def crear_funcion_1d(expr_str: str):
    """Crea f, f', f'' para una variable."""
    x = sp.Symbol("x")
    expr = sp.sympify(expr_str)
    f_prime = sp.diff(expr, x)
    f_double_prime = sp.diff(f_prime, x)

    f_lamb = sp.lambdify(x, expr, "numpy")
    fp_lamb = sp.lambdify(x, f_prime, "numpy")
    fdp_lamb = sp.lambdify(x, f_double_prime, "numpy")

    def f(v):
        return float(np.asarray(f_lamb(v)))

    def fp(v):
        return float(np.asarray(fp_lamb(v)))

    def fdp(v):
        return float(np.asarray(fdp_lamb(v)))

    return f, fp, fdp, expr, f_prime, f_double_prime


def crear_funcion_nd(expr_str: str, n: int):
    """Crea f, gradiente y Hessiana para n variables."""
    simbolos = variables_por_n(n)
    if n == 1:
        simbolos = (simbolos,)

    expr = sp.sympify(expr_str)
    grad_expr = [sp.diff(expr, s) for s in simbolos]
    hess_expr = [[sp.diff(g, s2) for s2 in simbolos] for g in grad_expr]

    f_lamb = sp.lambdify(simbolos, expr, "numpy")
    grad_lamb = [sp.lambdify(simbolos, g, "numpy") for g in grad_expr]
    hess_lamb = [[sp.lambdify(simbolos, h, "numpy") for h in fila] for fila in hess_expr]

    def f(x):
        x = np.array(x, dtype=float)
        return float(np.asarray(f_lamb(*x)))

    def grad(x):
        x = np.array(x, dtype=float)
        return np.array([float(np.asarray(g(*x))) for g in grad_lamb], dtype=float)

    def hess(x):
        x = np.array(x, dtype=float)
        H = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                H[i, j] = float(np.asarray(hess_lamb[i][j](*x)))
        return H

    return f, grad, hess, expr, grad_expr, hess_expr


# ============================================================
# ANÁLISIS MATEMÁTICO
# ============================================================

def clasificar_1d(f_double_prime, x):
    """Clasifica un punto crítico en 1D."""
    segunda = f_double_prime(x)
    if segunda > 1e-8:
        return "Mínimo local", segunda
    if segunda < -1e-8:
        return "Máximo local", segunda
    return "No concluyente con segunda derivada", segunda


def clasificar_hessiana(H):
    """Clasifica una matriz Hessiana por sus autovalores."""
    eigvals = np.linalg.eigvals(H)
    eigvals = np.real_if_close(eigvals)

    if np.all(eigvals > 1e-8):
        clasif = "Definida positiva: mínimo local estricto si ∇f=0."
    elif np.all(eigvals < -1e-8):
        clasif = "Definida negativa: máximo local estricto si ∇f=0."
    elif np.any(eigvals > 1e-8) and np.any(eigvals < -1e-8):
        clasif = "Indefinida: punto silla si ∇f=0."
    elif np.all(eigvals >= -1e-8):
        clasif = "Semidefinida positiva: posible mínimo, análisis no concluyente."
    elif np.all(eigvals <= 1e-8):
        clasif = "Semidefinida negativa: posible máximo, análisis no concluyente."
    else:
        clasif = "No concluyente."

    return clasif, eigvals


def analizar_kkt_lineal(x, grad, A_ineq, b_ineq, A_eq=None, b_eq=None, tipo="Minimizar", tol=1e-5):
    """
    Análisis KKT aproximado para restricciones lineales.
    Convención:
        A_ineq x <= b_ineq
        A_eq x = b_eq

    Para minimización:
        L = f(x) + λᵀ(Ax-b) + μᵀ(Aeqx-beq)
        λ >= 0 en restricciones activas.
    Para maximización se analiza minimizando -f.
    """
    x = np.array(x, dtype=float)
    g = np.array(grad(x), dtype=float)
    if tipo == "Maximizar":
        g = -g

    n = len(x)

    if A_ineq is None:
        A_ineq = np.empty((0, n))
        b_ineq = np.empty((0,))

    if A_eq is None:
        A_eq = np.empty((0, n))
        b_eq = np.empty((0,))

    slacks = b_ineq - A_ineq @ x if len(A_ineq) > 0 else np.empty((0,))
    activas = np.where(slacks <= tol)[0] if len(slacks) > 0 else np.array([], dtype=int)

    # Resolver g + A_act^T λ + A_eq^T μ = 0 por mínimos cuadrados
    matrices = []
    nombres = []

    if len(activas) > 0:
        matrices.append(A_ineq[activas].T)
        nombres += [f"lambda_{i+1}" for i in activas]

    if len(A_eq) > 0:
        matrices.append(A_eq.T)
        nombres += [f"mu_{i+1}" for i in range(len(A_eq))]

    resultado = {
        "factible": es_factible(x, A_ineq, b_ineq, A_eq, b_eq, tol=tol),
        "restricciones_activas": [int(i + 1) for i in activas],
        "slacks": slacks,
        "multiplicadores": {},
        "residuo_estacionariedad": None,
        "cumple_kkt_aprox": False,
        "comentario": "",
    }

    if len(matrices) == 0:
        residuo = np.linalg.norm(g)
        resultado["residuo_estacionariedad"] = residuo
        resultado["cumple_kkt_aprox"] = resultado["factible"] and residuo <= 1e-4
        resultado["comentario"] = "No hay restricciones activas; se verifica ∇f≈0."
        return resultado

    M = np.column_stack(matrices)

    try:
        mult, *_ = np.linalg.lstsq(M, -g, rcond=None)
        residuo = np.linalg.norm(g + M @ mult)

        for nombre, valor in zip(nombres, mult):
            resultado["multiplicadores"][nombre] = float(valor)

        lambdas = mult[:len(activas)] if len(activas) > 0 else np.empty((0,))
        dual_ok = np.all(lambdas >= -1e-5)
        resultado["residuo_estacionariedad"] = float(residuo)
        resultado["cumple_kkt_aprox"] = bool(resultado["factible"] and dual_ok and residuo <= 1e-4)

        if resultado["cumple_kkt_aprox"]:
            resultado["comentario"] = "Cumple condiciones KKT de forma aproximada."
        else:
            resultado["comentario"] = (
                "No cumple completamente KKT aproximado o el residuo es alto. "
                "Puede ser por no convexidad, tolerancias o convergencia local."
            )

    except Exception as exc:
        resultado["comentario"] = f"No se pudo calcular KKT aproximado: {exc}"

    return resultado


# ============================================================
# MÉTODOS PARA 1 VARIABLE
# ============================================================

def biseccion_derivada(f_prime, a, b, tol=1e-6, max_iter=100):
    """Bisección aplicada a f'(x)=0."""
    hist = []

    fa = f_prime(a)
    fb = f_prime(b)

    if abs(fa) < tol:
        return a, hist, "La raíz de f'(x) se encontró en a."
    if abs(fb) < tol:
        return b, hist, "La raíz de f'(x) se encontró en b."

    if fa * fb > 0:
        raise ValueError(
            "El intervalo no garantiza una raíz de f'(x)=0 porque f'(a) y f'(b) tienen el mismo signo. "
            "Pruebe otro intervalo."
        )

    c = (a + b) / 2

    for i in range(int(max_iter)):
        c = (a + b) / 2
        fc = f_prime(c)

        hist.append({
            "Iteración": i + 1,
            "a": a,
            "b": b,
            "c": c,
            "f'(a)": fa,
            "f'(b)": fb,
            "f'(c)": fc,
            "error": abs(b - a) / 2,
        })

        if abs(fc) <= tol or abs(b - a) / 2 <= tol:
            break

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, hist, "Bisección finalizada."


def newton_1d(f_prime, f_double_prime, x0, tol=1e-6, max_iter=100):
    """Newton para resolver f'(x)=0."""
    hist = []
    x = float(x0)

    for i in range(int(max_iter)):
        fp = f_prime(x)
        fdp = f_double_prime(x)

        hist.append({
            "Iteración": i + 1,
            "x": x,
            "f'(x)": fp,
            "f''(x)": fdp,
            "error |f'(x)|": abs(fp),
        })

        if abs(fp) <= tol:
            break

        if abs(fdp) <= 1e-12:
            raise ValueError(
                "La segunda derivada es cero o casi cero. Newton no puede continuar de forma segura."
            )

        x_new = x - fp / fdp

        if not np.isfinite(x_new):
            raise ValueError("Newton produjo un valor no finito. Revise la función o el punto inicial.")

        if abs(x_new - x) <= tol:
            x = x_new
            break

        x = x_new

    return x, hist, "Newton finalizado."


def optimizacion_1d_robusta(f, a, b, tipo="Minimizar"):
    """Optimización robusta en intervalo usando scipy.optimize.minimize_scalar."""
    if a >= b:
        raise ValueError("El extremo inferior a debe ser menor que b.")

    objetivo = f if tipo == "Minimizar" else lambda x: -f(x)

    res = opt.minimize_scalar(
        objetivo,
        bounds=(a, b),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )

    if not res.success:
        raise ValueError("SciPy no pudo resolver la optimización 1D en el intervalo.")

    x_opt = float(res.x)
    f_opt = f(x_opt)

    # Comparar también contra los bordes porque el óptimo global en un intervalo puede estar en a o b.
    candidatos = [
        {"Punto": "Interior/SciPy", "x": x_opt, "f(x)": f_opt},
        {"Punto": "Borde a", "x": float(a), "f(x)": f(float(a))},
        {"Punto": "Borde b", "x": float(b), "f(x)": f(float(b))},
    ]

    if tipo == "Minimizar":
        mejor = min(candidatos, key=lambda r: r["f(x)"])
    else:
        mejor = max(candidatos, key=lambda r: r["f(x)"])

    return mejor["x"], mejor["f(x)"], candidatos, res


# ============================================================
# MÉTODOS PARA VARIAS VARIABLES
# ============================================================

def gradiente_iterativo(f, grad, x0, alpha=0.1, tol=1e-6, max_iter=100, tipo="Minimizar"):
    """Descenso/ascenso por gradiente con paso fijo."""
    x = np.array(x0, dtype=float)
    hist = []
    signo = -1 if tipo == "Minimizar" else 1

    for i in range(int(max_iter)):
        g = grad(x)
        norm_g = np.linalg.norm(g)

        fila = {"Iteración": i + 1}
        for j, val in enumerate(x):
            fila[f"x{j+1}"] = val
        fila["f(x)"] = f(x)
        fila["||grad||"] = norm_g
        hist.append(fila)

        if norm_g <= tol:
            break

        x_new = x + signo * alpha * g

        if not np.all(np.isfinite(x_new)):
            raise ValueError("El método de gradiente produjo valores no finitos.")

        if np.linalg.norm(x_new - x) <= tol:
            x = x_new
            break

        x = x_new

    return x, hist


def newton_nd_iterativo(f, grad, hess, x0, tol=1e-6, max_iter=50, tipo="Minimizar"):
    """
    Newton multivariable con amortiguamiento.
    Para maximizar se aplica Newton sobre -f.
    """
    x = np.array(x0, dtype=float)
    hist = []

    for i in range(int(max_iter)):
        g_original = grad(x)
        H_original = hess(x)

        if tipo == "Minimizar":
            g = g_original
            H = H_original
            valor_obj = f(x)
        else:
            g = -g_original
            H = -H_original
            valor_obj = -f(x)

        norm_g = np.linalg.norm(g_original)

        fila = {"Iteración": i + 1}
        for j, val in enumerate(x):
            fila[f"x{j+1}"] = val
        fila["f(x)"] = f(x)
        fila["||grad||"] = norm_g
        hist.append(fila)

        if norm_g <= tol:
            break

        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -np.linalg.pinv(H) @ g

        # Búsqueda amortiguada para evitar saltos explosivos.
        paso = 1.0
        objetivo_actual = valor_obj

        for _ in range(25):
            x_trial = x + paso * p

            if not np.all(np.isfinite(x_trial)):
                paso *= 0.5
                continue

            if tipo == "Minimizar":
                objetivo_trial = f(x_trial)
            else:
                objetivo_trial = -f(x_trial)

            if objetivo_trial <= objetivo_actual + 1e-4 * paso * np.dot(g, p):
                break

            paso *= 0.5

        x_new = x + paso * p

        if np.linalg.norm(x_new - x) <= tol:
            x = x_new
            break

        x = x_new

    return x, hist


def scipy_no_restringida(f, grad, hess, x0, tipo="Minimizar"):
    """Optimización local robusta no restringida usando SciPy."""
    x0 = np.array(x0, dtype=float)

    if tipo == "Minimizar":
        fun = f
        jac = grad
        hess_fun = hess
    else:
        fun = lambda x: -f(x)
        jac = lambda x: -grad(x)
        hess_fun = lambda x: -hess(x)

    res = opt.minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        hess=hess_fun,
        method="trust-exact",
        options={"gtol": 1e-10, "maxiter": 1000},
    )

    # Fallback por si trust-exact falla con alguna Hessiana complicada.
    if not res.success:
        res = opt.minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 1000},
        )

    return res.x, f(res.x), res


# ============================================================
# MÉTODOS CON RESTRICCIONES LINEALES
# ============================================================

def scipy_restringida_lineal(f, grad, x0, A_ineq, b_ineq, A_eq=None, b_eq=None, tipo="Minimizar"):
    """Optimización no lineal con restricciones lineales usando SLSQP."""
    x0 = np.array(x0, dtype=float)

    if not es_factible(x0, A_ineq, b_ineq, A_eq, b_eq):
        x0 = proyectar_punto(x0, A_ineq, b_ineq, A_eq, b_eq)

    if tipo == "Minimizar":
        fun = f
        jac = grad
    else:
        fun = lambda x: -f(x)
        jac = lambda x: -grad(x)

    cons = restricciones_scipy(A_ineq, b_ineq, A_eq, b_eq)

    res = opt.minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )

    if not res.success:
        st.warning(
            "SciPy terminó con advertencia: "
            + str(res.message)
            + ". Revise si el problema es no convexo o si las restricciones son inconsistentes."
        )

    return res.x, f(res.x), res


def gradiente_proyectado(f, grad, x0, A_ineq, b_ineq, A_eq=None, b_eq=None,
                         alpha=0.1, tol=1e-6, max_iter=100, tipo="Minimizar"):
    """Gradiente proyectado para restricciones lineales."""
    x = proyectar_punto(x0, A_ineq, b_ineq, A_eq, b_eq)
    hist = []
    signo = -1 if tipo == "Minimizar" else 1

    for i in range(int(max_iter)):
        g = grad(x)

        fila = {"Iteración": i + 1}
        for j, val in enumerate(x):
            fila[f"x{j+1}"] = val
        fila["f(x)"] = f(x)
        fila["||grad||"] = np.linalg.norm(g)
        hist.append(fila)

        y = x + signo * alpha * g
        x_new = proyectar_punto(y, A_ineq, b_ineq, A_eq, b_eq)

        if np.linalg.norm(x_new - x) <= tol:
            x = x_new
            break

        x = x_new

    return x, hist


def vertices_factibles(A_ineq, b_ineq, A_eq=None, b_eq=None, n=2):
    """
    Calcula vértices de una región lineal.
    Importante: en PNL, evaluar vértices NO garantiza óptimo general.
    Se usa como análisis de candidatos geométricos.
    """
    if A_ineq is None:
        A_ineq = np.empty((0, n))
        b_ineq = np.empty((0,))

    if A_eq is None:
        A_eq = np.empty((0, n))
        b_eq = np.empty((0,))

    vertices = []

    # Se combinan restricciones activas hasta completar n ecuaciones.
    indices_ineq = list(range(len(A_ineq)))
    needed_from_ineq = n - len(A_eq)

    if needed_from_ineq < 0:
        return []

    for comb in itertools.combinations(indices_ineq, needed_from_ineq):
        A_rows = []
        b_rows = []

        if len(A_eq) > 0:
            A_rows.extend(A_eq.tolist())
            b_rows.extend(b_eq.tolist())

        for idx in comb:
            A_rows.append(A_ineq[idx].tolist())
            b_rows.append(float(b_ineq[idx]))

        A_sub = np.array(A_rows, dtype=float)
        b_sub = np.array(b_rows, dtype=float)

        if A_sub.shape[0] == n and np.linalg.matrix_rank(A_sub) == n:
            try:
                x = np.linalg.solve(A_sub, b_sub)
                if es_factible(x, A_ineq, b_ineq, A_eq, b_eq, tol=1e-6):
                    vertices.append(x)
            except np.linalg.LinAlgError:
                pass

    # Eliminar duplicados
    unicos = []
    for v in vertices:
        if not any(np.linalg.norm(v - u) <= 1e-6 for u in unicos):
            unicos.append(v)

    return unicos


# ============================================================
# OPTIMIZACIÓN CUADRÁTICA
# ============================================================

def funcion_cuadratica(Q, c, k):
    """Crea f, gradiente y Hessiana para 1/2 xTQx + cTx + k."""
    Q = np.array(Q, dtype=float)
    Q = (Q + Q.T) / 2
    c = np.array(c, dtype=float)

    def f(x):
        x = np.array(x, dtype=float)
        return float(0.5 * x.T @ Q @ x + c.T @ x + k)

    def grad(x):
        x = np.array(x, dtype=float)
        return Q @ x + c

    def hess(x):
        return Q

    return f, grad, hess, Q


def resolver_cuadratica(Q, c, k, x0, A_ineq, b_ineq, A_eq=None, b_eq=None, tipo="Minimizar"):
    """Resuelve programación cuadrática con restricciones lineales."""
    f, grad, hess, Qsym = funcion_cuadratica(Q, c, k)
    x_opt, f_opt, res = scipy_restringida_lineal(
        f=f,
        grad=grad,
        x0=x0,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        tipo=tipo,
    )
    return x_opt, f_opt, res, f, grad, hess, Qsym


# ============================================================
# GRÁFICAS
# ============================================================

def graficar_1d(f, x_opt, a, b):
    """Gráfica 1D de la función."""
    margen = max(1.0, abs(b - a) * 0.15)
    xs = np.linspace(a - margen, b + margen, 500)
    ys = []

    for xi in xs:
        try:
            ys.append(f(xi))
        except Exception:
            ys.append(np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="f(x)",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_opt],
            y=[f(x_opt)],
            mode="markers",
            name="Punto óptimo/candidato",
            marker=dict(size=11, symbol="star"),
        )
    )
    fig.update_layout(
        title="Gráfica de f(x)",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=450,
    )
    return fig


def rango_grafica_2d(x_opt, vertices=None):
    """Calcula rangos razonables para la gráfica 2D."""
    puntos = [np.array(x_opt, dtype=float)]

    if vertices:
        puntos += [np.array(v, dtype=float) for v in vertices]

    puntos = np.array(puntos)

    xmin, ymin = puntos.min(axis=0)
    xmax, ymax = puntos.max(axis=0)

    if abs(xmax - xmin) < 1:
        xmin -= 5
        xmax += 5
    else:
        margen_x = 0.25 * abs(xmax - xmin)
        xmin -= margen_x
        xmax += margen_x

    if abs(ymax - ymin) < 1:
        ymin -= 5
        ymax += 5
    else:
        margen_y = 0.25 * abs(ymax - ymin)
        ymin -= margen_y
        ymax += margen_y

    return xmin, xmax, ymin, ymax


def graficar_2d(f, x_opt, hist=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, vertices=None):
    """Curvas de nivel para problemas de 2 variables."""
    x_opt = np.array(x_opt, dtype=float)

    xmin, xmax, ymin, ymax = rango_grafica_2d(x_opt, vertices)

    x_range = np.linspace(xmin, xmax, 180)
    y_range = np.linspace(ymin, ymax, 180)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.empty_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            punto = np.array([X[i, j], Y[i, j]], dtype=float)
            if A_ineq is not None and len(A_ineq) > 0:
                if not es_factible(punto, A_ineq, b_ineq, A_eq, b_eq, tol=1e-8):
                    Z[i, j] = np.nan
                    continue
            try:
                Z[i, j] = f(punto)
            except Exception:
                Z[i, j] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            contours_coloring="lines",
            name="Curvas de nivel",
            showscale=True,
        )
    )

    if hist:
        hx = []
        hy = []
        for h in hist:
            if "x1" in h and "x2" in h:
                hx.append(h["x1"])
                hy.append(h["x2"])
        if hx and hy:
            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=hy,
                    mode="lines+markers",
                    name="Iteraciones",
                    marker=dict(size=6),
                )
            )

    if vertices:
        vx = [v[0] for v in vertices]
        vy = [v[1] for v in vertices]
        fig.add_trace(
            go.Scatter(
                x=vx,
                y=vy,
                mode="markers",
                name="Vértices factibles",
                marker=dict(size=8, symbol="circle"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[x_opt[0]],
            y=[x_opt[1]],
            mode="markers",
            name="Óptimo/candidato",
            marker=dict(size=13, symbol="star"),
        )
    )

    # Dibujar rectas de restricciones de desigualdad.
    if A_ineq is not None and len(A_ineq) > 0:
        xs = np.linspace(xmin, xmax, 300)
        for idx, (a, b) in enumerate(zip(A_ineq, b_ineq), start=1):
            if abs(a[1]) > 1e-12:
                ys = (b - a[0] * xs) / a[1]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"Restricción {idx}",
                        line=dict(dash="dash"),
                    )
                )
            elif abs(a[0]) > 1e-12:
                xval = b / a[0]
                fig.add_trace(
                    go.Scatter(
                        x=[xval, xval],
                        y=[ymin, ymax],
                        mode="lines",
                        name=f"Restricción {idx}",
                        line=dict(dash="dash"),
                    )
                )

    fig.update_layout(
        title="Curvas de nivel y región factible",
        xaxis_title="x1",
        yaxis_title="x2",
        height=560,
    )
    return fig


# ============================================================
# COMPONENTES DE INTERFAZ
# ============================================================

def mostrar_tabla_resultado(x_opt, f_opt, nombre_x="x óptimo"):
    """Muestra un resumen numérico."""
    st.markdown("### Resultado principal")
    st.write(f"**{nombre_x}:** `{np.round(x_opt, 8)}`")
    st.write(f"**Valor de la función:** `{f_opt:.10f}`")


def mostrar_hessiana(H):
    """Muestra matriz Hessiana y clasificación."""
    clasif, eigvals = clasificar_hessiana(H)
    st.markdown("### Análisis de Hessiana")
    st.write("**Matriz Hessiana:**")
    st.dataframe(pd.DataFrame(np.round(H, 8)), use_container_width=True)
    st.write(f"**Autovalores:** `{np.round(eigvals, 8)}`")
    st.info(clasif)


def mostrar_kkt(kkt):
    """Muestra análisis KKT aproximado."""
    st.markdown("### Análisis KKT aproximado")
    st.write(f"**Factible:** `{kkt['factible']}`")
    st.write(f"**Restricciones activas:** `{kkt['restricciones_activas']}`")
    st.write(f"**Residuo de estacionariedad:** `{kkt['residuo_estacionariedad']}`")
    st.write(f"**Cumple KKT aproximado:** `{kkt['cumple_kkt_aprox']}`")
    st.write(f"**Comentario:** {kkt['comentario']}")

    if len(kkt["slacks"]) > 0:
        st.write("**Holguras b - Ax:**")
        st.dataframe(pd.DataFrame({"Restricción": range(1, len(kkt["slacks"]) + 1), "Holgura": kkt["slacks"]}))

    if kkt["multiplicadores"]:
        st.write("**Multiplicadores aproximados:**")
        st.dataframe(pd.DataFrame(list(kkt["multiplicadores"].items()), columns=["Multiplicador", "Valor"]))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Menú")
opcion = st.sidebar.radio(
    "Seleccione el módulo:",
    [
        "1. Una variable",
        "2. Varias variables sin restricciones",
        "3. No lineal con restricciones lineales",
        "4. Programación cuadrática",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Recomendación")
st.sidebar.info(
    "Use expresiones con sintaxis de Python/SymPy. Ejemplos: "
    "`x**2`, `sin(x)`, `x1**2 + x2**2`, `exp(x1)`."
)


# ============================================================
# MÓDULO 1: UNA VARIABLE
# ============================================================

if opcion.startswith("1"):
    st.markdown("## 1. Optimización no restringida de una variable")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("### Datos de entrada")

        expr_str = st.text_input(
            "Función f(x):",
            value="x**3 - 6*x**2 + 9*x + 1",
        )

        metodo = st.selectbox(
            "Método de cálculo:",
            [
                "SciPy robusto en intervalo",
                "Bisección sobre f'(x)",
                "Newton sobre f'(x)",
            ],
        )

        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"])

        c1, c2 = st.columns(2)
        a = c1.number_input("a:", value=0.0)
        b = c2.number_input("b:", value=3.0)

        x0 = st.number_input("x0 para Newton:", value=1.5)

        c3, c4 = st.columns(2)
        tol = c3.number_input("Tolerancia:", value=0.000001, format="%f")
        max_iter = c4.number_input("Iteraciones máximas:", value=100, step=1)

        resolver = st.button("Resolver módulo 1")

    with col2:
        if resolver:
            try:
                f, fp, fdp, expr, expr_p, expr_dp = crear_funcion_1d(expr_str)

                hist = []
                candidatos = []

                if metodo == "SciPy robusto en intervalo":
                    x_opt, f_opt, candidatos, res = optimizacion_1d_robusta(f, a, b, tipo)
                    mensaje = "Optimización robusta en intervalo finalizada."
                elif metodo == "Bisección sobre f'(x)":
                    x_opt, hist, mensaje = biseccion_derivada(fp, a, b, tol, max_iter)
                    f_opt = f(x_opt)
                else:
                    x_opt, hist, mensaje = newton_1d(fp, fdp, x0, tol, max_iter)
                    f_opt = f(x_opt)

                clasificacion, segunda = clasificar_1d(fdp, x_opt)

                st.success(mensaje)

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Respuesta final", "Derivadas y análisis", "Iteraciones/candidatos", "Gráfica"]
                )

                with tab1:
                    st.markdown("### Resultado")
                    st.write(f"**Método:** {metodo}")
                    st.write(f"**Objetivo:** {tipo}")
                    st.write(f"**x encontrado:** `{x_opt:.10f}`")
                    st.write(f"**f(x):** `{f_opt:.10f}`")
                    st.write(f"**Clasificación local:** {clasificacion}")

                    if metodo != "SciPy robusto en intervalo":
                        st.warning(
                            "Este método encuentra un punto crítico local. "
                            "Para óptimo global en intervalo, use el método SciPy robusto y compare bordes."
                        )

                with tab2:
                    st.write(f"**f(x) =** `{sp.sstr(expr)}`")
                    st.write(f"**f'(x) =** `{sp.sstr(expr_p)}`")
                    st.write(f"**f''(x) =** `{sp.sstr(expr_dp)}`")
                    st.write(f"**f''(x encontrado) =** `{segunda:.10f}`")

                with tab3:
                    if hist:
                        st.dataframe(pd.DataFrame(hist), use_container_width=True)
                    if candidatos:
                        st.write("**Candidatos evaluados:**")
                        st.dataframe(pd.DataFrame(candidatos), use_container_width=True)

                with tab4:
                    st.plotly_chart(graficar_1d(f, x_opt, a, b), use_container_width=True)

            except Exception as exc:
                mostrar_error_amigable("No se pudo resolver el módulo 1.", exc)


# ============================================================
# MÓDULO 2: VARIAS VARIABLES SIN RESTRICCIONES
# ============================================================

elif opcion.startswith("2"):
    st.markdown("## 2. Optimización no restringida de varias variables")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("### Datos de entrada")

        n = st.selectbox("Número de variables:", [2, 3, 4, 5], index=0)

        default_expr = (
            "x1**2 + x2**2 - 4*x1 - 6*x2 + 13"
            if n == 2
            else " + ".join([f"x{i+1}**2" for i in range(n)])
        )

        expr_str = st.text_input("Función f(x):", value=default_expr)

        metodo = st.selectbox(
            "Método:",
            [
                "SciPy robusto local",
                "Gradiente paso fijo",
                "Newton amortiguado",
            ],
        )

        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"])

        p0_default = ",".join(["0"] * n)
        p0_str = st.text_input("Punto inicial:", value=p0_default)

        alpha = st.number_input("Alpha para gradiente:", value=0.1)

        c1, c2 = st.columns(2)
        tol = c1.number_input("Tolerancia:", value=0.000001, format="%f")
        max_iter = c2.number_input("Iteraciones máximas:", value=100, step=1)

        resolver = st.button("Resolver módulo 2")

    with col2:
        if resolver:
            try:
                x0 = parse_vector(p0_str)
                validar_dimension_vector(x0, n, "punto inicial")

                f, grad, hess, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, n)

                hist = []

                if metodo == "SciPy robusto local":
                    x_opt, f_opt, res = scipy_no_restringida(f, grad, hess, x0, tipo)
                    mensaje = "SciPy finalizó la optimización local."
                    if not res.success:
                        st.warning(f"Advertencia de SciPy: {res.message}")
                elif metodo == "Gradiente paso fijo":
                    x_opt, hist = gradiente_iterativo(f, grad, x0, alpha, tol, max_iter, tipo)
                    f_opt = f(x_opt)
                    mensaje = "Método de gradiente finalizado."
                else:
                    x_opt, hist = newton_nd_iterativo(f, grad, hess, x0, tol, max_iter, tipo)
                    f_opt = f(x_opt)
                    mensaje = "Método de Newton amortiguado finalizado."

                H = hess(x_opt)
                g = grad(x_opt)

                st.success(mensaje)

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Respuesta final", "Análisis matemático", "Iteraciones", "Gráfica"]
                )

                with tab1:
                    mostrar_tabla_resultado(x_opt, f_opt)
                    st.write(f"**Norma del gradiente:** `{np.linalg.norm(g):.10f}`")
                    st.write(f"**Objetivo:** {tipo}")

                    if tipo == "Maximizar":
                        st.info(
                            "Para maximización, internamente se optimiza -f cuando se usa SciPy/Newton robusto."
                        )

                with tab2:
                    st.write(f"**f(x) =** `{sp.sstr(expr)}`")
                    st.write("**Gradiente:**")
                    for i, ge in enumerate(grad_expr, start=1):
                        st.write(f"∂f/∂x{i} = `{sp.sstr(ge)}`")
                    mostrar_hessiana(H)

                with tab3:
                    if hist:
                        st.dataframe(pd.DataFrame(hist), use_container_width=True)
                    else:
                        st.info("El método SciPy no muestra tabla manual de iteraciones en esta versión.")

                with tab4:
                    if n == 2:
                        st.plotly_chart(graficar_2d(f, x_opt, hist), use_container_width=True)
                    else:
                        st.info("La gráfica de curvas de nivel solo está disponible para 2 variables.")

            except Exception as exc:
                mostrar_error_amigable("No se pudo resolver el módulo 2.", exc)


# ============================================================
# MÓDULO 3: NO LINEAL CON RESTRICCIONES LINEALES
# ============================================================

elif opcion.startswith("3"):
    st.markdown("## 3. Optimización no lineal con restricciones lineales")

    st.warning(
        "Importante: en programación no lineal, evaluar únicamente vértices NO garantiza el óptimo global. "
        "El método principal recomendado aquí es SLSQP de SciPy, complementado con KKT aproximado."
    )

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("### Datos de entrada")

        n = st.selectbox("Número de variables:", [2, 3, 4, 5], index=0)

        expr_default = "x1*x2 + 2*x1 + x2" if n == 2 else " + ".join([f"x{i+1}**2" for i in range(n)])
        expr_str = st.text_input("Función f(x):", value=expr_default)

        metodo = st.selectbox(
            "Método:",
            [
                "SciPy SLSQP con restricciones",
                "Gradiente proyectado",
                "Solo analizar vértices factibles",
            ],
        )

        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"], index=1)

        st.markdown("Restricciones: `a1,a2,...,b,signo`")
        st.caption("Ejemplo para 2 variables: `1,0,4,<=` significa x1 <= 4.")

        restr_default = (
            "1,0,4,<=\n"
            "0,1,6,<=\n"
            "3,2,18,<=\n"
            "1,0,0,>=\n"
            "0,1,0,>="
        )

        restricciones_str = st.text_area("Restricciones:", value=restr_default, height=150)

        p0_default = ",".join(["1"] * n)
        p0_str = st.text_input("Punto inicial:", value=p0_default)

        alpha = st.number_input("Alpha para gradiente proyectado:", value=0.1)

        c1, c2 = st.columns(2)
        tol = c1.number_input("Tolerancia:", value=0.000001, format="%f")
        max_iter = c2.number_input("Iteraciones máximas:", value=100, step=1)

        resolver = st.button("Resolver módulo 3")

    with col2:
        if resolver:
            try:
                x0 = parse_vector(p0_str)
                validar_dimension_vector(x0, n, "punto inicial")

                A_ineq, b_ineq, A_eq, b_eq = parse_restrictions(restricciones_str, n)

                f, grad, hess, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, n)

                hist = []
                vertices = vertices_factibles(A_ineq, b_ineq, A_eq, b_eq, n=n)
                tabla_vertices = []

                for i, v in enumerate(vertices, start=1):
                    tabla_vertices.append({
                        "Vértice": i,
                        **{f"x{j+1}": v[j] for j in range(n)},
                        "f(x)": f(v),
                    })

                if metodo == "SciPy SLSQP con restricciones":
                    x_opt, f_opt, res = scipy_restringida_lineal(
                        f, grad, x0, A_ineq, b_ineq, A_eq, b_eq, tipo
                    )
                    mensaje = "Optimización restringida con SciPy SLSQP finalizada."
                elif metodo == "Gradiente proyectado":
                    x_opt, hist = gradiente_proyectado(
                        f, grad, x0, A_ineq, b_ineq, A_eq, b_eq,
                        alpha=alpha, tol=tol, max_iter=max_iter, tipo=tipo
                    )
                    f_opt = f(x_opt)
                    mensaje = "Gradiente proyectado finalizado."
                else:
                    if not tabla_vertices:
                        raise ValueError("No se encontraron vértices factibles.")
                    if tipo == "Minimizar":
                        mejor = min(tabla_vertices, key=lambda r: r["f(x)"])
                    else:
                        mejor = max(tabla_vertices, key=lambda r: r["f(x)"])
                    x_opt = np.array([mejor[f"x{j+1}"] for j in range(n)], dtype=float)
                    f_opt = f(x_opt)
                    mensaje = (
                        "Análisis de vértices finalizado. Recuerde: esto no garantiza óptimo "
                        "en programación no lineal general."
                    )

                H = hess(x_opt)
                kkt = analizar_kkt_lineal(x_opt, grad, A_ineq, b_ineq, A_eq, b_eq, tipo=tipo)

                st.success(mensaje)

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["Respuesta final", "Restricciones/KKT", "Vértices", "Iteraciones", "Gráfica"]
                )

                with tab1:
                    mostrar_tabla_resultado(x_opt, f_opt)
                    st.write(f"**Factible:** `{es_factible(x_opt, A_ineq, b_ineq, A_eq, b_eq)}`")
                    st.write(f"**Norma del gradiente:** `{np.linalg.norm(grad(x_opt)):.10f}`")
                    mostrar_hessiana(H)

                with tab2:
                    st.write("**Matriz de desigualdades A x <= b:**")
                    if len(A_ineq) > 0:
                        st.dataframe(pd.DataFrame(A_ineq), use_container_width=True)
                        st.write("**b:**")
                        st.dataframe(pd.DataFrame(b_ineq, columns=["b"]), use_container_width=True)
                    else:
                        st.info("No se ingresaron desigualdades.")

                    if len(A_eq) > 0:
                        st.write("**Matriz de igualdades Aeq x = beq:**")
                        st.dataframe(pd.DataFrame(A_eq), use_container_width=True)
                        st.dataframe(pd.DataFrame(b_eq, columns=["beq"]), use_container_width=True)

                    mostrar_kkt(kkt)

                with tab3:
                    if tabla_vertices:
                        st.dataframe(pd.DataFrame(tabla_vertices), use_container_width=True)
                    else:
                        st.info("No se encontraron vértices factibles o no aplican para esta geometría.")

                with tab4:
                    if hist:
                        st.dataframe(pd.DataFrame(hist), use_container_width=True)
                    else:
                        st.info("No hay iteraciones manuales para este método.")

                with tab5:
                    if n == 2:
                        st.plotly_chart(
                            graficar_2d(
                                f,
                                x_opt,
                                hist=hist,
                                A_ineq=A_ineq,
                                b_ineq=b_ineq,
                                A_eq=A_eq,
                                b_eq=b_eq,
                                vertices=vertices,
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("La gráfica solo está disponible para 2 variables.")

            except Exception as exc:
                mostrar_error_amigable("No se pudo resolver el módulo 3.", exc)


# ============================================================
# MÓDULO 4: PROGRAMACIÓN CUADRÁTICA
# ============================================================

elif opcion.startswith("4"):
    st.markdown("## 4. Programación cuadrática con restricciones lineales")

    st.markdown(
        "Modelo:  \n"
        "$$f(x)=\\frac{1}{2}x^TQx+c^Tx+k$$"
    )

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("### Datos de entrada")

        n = st.selectbox("Número de variables:", [2, 3, 4, 5], index=0)

        if n == 2:
            Q_default = "2,0\n0,2"
            c_default = "-4,-6"
            restr_default = (
                "1,0,4,<=\n"
                "0,1,6,<=\n"
                "3,2,18,<=\n"
                "1,0,0,>=\n"
                "0,1,0,>="
            )
            x0_default = "0,0"
        else:
            Q_default = "\n".join(
                ",".join(["2" if i == j else "0" for j in range(n)])
                for i in range(n)
            )
            c_default = ",".join(["-1"] * n)
            restr_default = "\n".join(
                [
                    ",".join(["1" if i == j else "0" for i in range(n)] + ["10", "<="])
                    for j in range(n)
                ]
                + [
                    ",".join(["1" if i == j else "0" for i in range(n)] + ["0", ">="])
                    for j in range(n)
                ]
            )
            x0_default = ",".join(["1"] * n)

        Q_str = st.text_area("Matriz Q:", value=Q_default, height=140)
        c_str = st.text_input("Vector c:", value=c_default)
        k = st.number_input("Constante k:", value=13.0)

        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"])

        st.markdown("Restricciones: `a1,a2,...,b,signo`")
        restricciones_str = st.text_area("Restricciones:", value=restr_default, height=150)

        x0_str = st.text_input("Punto inicial:", value=x0_default)

        resolver = st.button("Resolver módulo 4")

    with col2:
        if resolver:
            try:
                Q = parse_matrix(Q_str)
                c = parse_vector(c_str)
                x0 = parse_vector(x0_str)

                validar_dimension_matriz(Q, n, n, "matriz Q")
                validar_dimension_vector(c, n, "vector c")
                validar_dimension_vector(x0, n, "punto inicial")

                A_ineq, b_ineq, A_eq, b_eq = parse_restrictions(restricciones_str, n)

                x_opt, f_opt, res, f, grad, hess, Qsym = resolver_cuadratica(
                    Q, c, k, x0, A_ineq, b_ineq, A_eq, b_eq, tipo
                )

                clasif_Q, eigvals_Q = clasificar_hessiana(Qsym)
                kkt = analizar_kkt_lineal(x_opt, grad, A_ineq, b_ineq, A_eq, b_eq, tipo=tipo)

                vertices = vertices_factibles(A_ineq, b_ineq, A_eq, b_eq, n=n)

                st.success("Programación cuadrática resuelta.")

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Respuesta final", "Análisis de Q", "KKT/restricciones", "Gráfica"]
                )

                with tab1:
                    mostrar_tabla_resultado(x_opt, f_opt)
                    st.write(f"**Factible:** `{es_factible(x_opt, A_ineq, b_ineq, A_eq, b_eq)}`")
                    st.write(f"**Mensaje de SciPy:** `{res.message}`")

                with tab2:
                    st.write("**Q simétrica usada:**")
                    st.dataframe(pd.DataFrame(np.round(Qsym, 8)), use_container_width=True)
                    st.write(f"**Autovalores de Q:** `{np.round(eigvals_Q, 8)}`")
                    st.info(clasif_Q)

                    if tipo == "Minimizar" and np.all(eigvals_Q > 1e-8):
                        st.success("El problema es convexo si las restricciones son lineales. El mínimo local es global.")
                    elif tipo == "Maximizar" and np.all(eigvals_Q < -1e-8):
                        st.success("El problema es cóncavo si las restricciones son lineales. El máximo local es global.")
                    else:
                        st.warning(
                            "No hay garantía general de óptimo global por convexidad/concavidad. "
                            "Se recomienda revisar KKT y candidatos."
                        )

                with tab3:
                    mostrar_kkt(kkt)

                    st.write("**Restricciones de desigualdad A x <= b:**")
                    if len(A_ineq) > 0:
                        st.dataframe(pd.DataFrame(A_ineq), use_container_width=True)
                        st.dataframe(pd.DataFrame(b_ineq, columns=["b"]), use_container_width=True)
                    else:
                        st.info("No se ingresaron restricciones de desigualdad.")

                    if len(A_eq) > 0:
                        st.write("**Restricciones de igualdad:**")
                        st.dataframe(pd.DataFrame(A_eq), use_container_width=True)
                        st.dataframe(pd.DataFrame(b_eq, columns=["beq"]), use_container_width=True)

                with tab4:
                    if n == 2:
                        st.plotly_chart(
                            graficar_2d(
                                f,
                                x_opt,
                                hist=[],
                                A_ineq=A_ineq,
                                b_ineq=b_ineq,
                                A_eq=A_eq,
                                b_eq=b_eq,
                                vertices=vertices,
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("La gráfica solo está disponible para 2 variables.")

            except Exception as exc:
                mostrar_error_amigable("No se pudo resolver el módulo 4.", exc)


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("---")
st.caption(
    "Aplicación académica para Programación No Lineal. "
    "Incluye métodos numéricos locales; en problemas no convexos los resultados deben interpretarse con criterio matemático."
)

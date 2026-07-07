import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import itertools
import re

# ============================================================
# CONFIGURACION GENERAL
# ============================================================
st.set_page_config(
    page_title="Programacion No Lineal | UCuenca",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #eef5ff 0%, #f8fbff 45%, #ffffff 100%);
    }
    .main-card {
        background: linear-gradient(135deg, #08203a 0%, #0f4c81 100%);
        padding: 24px 28px;
        border-radius: 22px;
        color: white;
        box-shadow: 0 10px 28px rgba(8, 32, 58, 0.18);
        margin-bottom: 18px;
    }
    .main-card h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: white;
    }
    .main-card h3 {
        margin: 4px 0 8px 0;
        color: #b9d8ff;
        font-weight: 500;
    }
    .main-card p {
        margin: 0;
        color: #e5f0ff;
    }
    .info-card {
        background: white;
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #dce8f7;
        box-shadow: 0 6px 18px rgba(23, 53, 85, 0.08);
        margin-bottom: 14px;
    }
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #edf6ff 100%);
        border: 1px solid #dce8f7;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(23, 53, 85, 0.06);
        margin-bottom: 10px;
    }
    .metric-title {
        color: #57708f;
        font-size: 0.84rem;
        margin-bottom: 3px;
    }
    .metric-value {
        color: #08203a;
        font-size: 1.15rem;
        font-weight: 750;
    }
    .stButton>button {
        background: linear-gradient(135deg, #008C7A 0%, #00A98F 100%);
        color: white;
        font-weight: 800;
        border-radius: 12px;
        border: none;
        width: 100%;
        padding: 0.7rem 1rem;
        box-shadow: 0 6px 16px rgba(0, 140, 122, 0.22);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #006B5D 0%, #008C7A 100%);
        color: white;
    }
    div[data-testid="stSidebar"] {
        background: #071d33;
    }
    div[data-testid="stSidebar"] * {
        color: #eef7ff;
    }
    div[data-testid="stSidebar"] .stRadio label {
        color: #eef7ff !important;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-card">
        <h1>PROGRAMACION NO LINEAL</h1>
        <h3>Universidad de Cuenca</h3>
        <p><b>Materia:</b> Investigacion de Operaciones | <b>Carrera:</b> Ingenieria en Telecomunicaciones</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# UTILIDADES GENERALES
# ============================================================
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


def simbolos_n(n: int):
    return tuple(sp.Symbol(f"x{i+1}", real=True) for i in range(n))


def texto_limpio(texto: str) -> str:
    """Permite escribir ^ en lugar de ** y elimina espacios extremos."""
    return texto.strip().replace("^", "**")


def parsear_expresion(expr_str: str, n: int):
    symbols = simbolos_n(n)
    local_dict = {f"x{i+1}": symbols[i] for i in range(n)}
    local_dict.update({"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
    return parse_expr(texto_limpio(expr_str), local_dict=local_dict, transformations=TRANSFORMATIONS)


def crear_funcion_nd(expr_str: str, n: int):
    symbols = simbolos_n(n)
    expr = parsear_expresion(expr_str, n)
    grad_expr = [sp.diff(expr, s) for s in symbols]
    hess_expr = sp.Matrix([[sp.diff(g, s2) for s2 in symbols] for g in grad_expr])

    f_lamb = sp.lambdify(symbols, expr, "numpy")
    grad_lamb = [sp.lambdify(symbols, g, "numpy") for g in grad_expr]
    hess_lamb = sp.lambdify(symbols, hess_expr, "numpy")

    def f(x):
        x = np.array(x, dtype=float)
        return float(f_lamb(*x))

    def grad(x):
        x = np.array(x, dtype=float)
        return np.array([float(g(*x)) for g in grad_lamb], dtype=float)

    def hess(x):
        x = np.array(x, dtype=float)
        return np.array(hess_lamb(*x), dtype=float)

    return f, grad, hess, expr, grad_expr, hess_expr


def crear_funcion_1d(expr_str: str):
    f, grad, hess, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, 1)

    def fp(x):
        return float(grad([x])[0])

    def fpp(x):
        return float(hess([x])[0, 0])

    def f1(x):
        return float(f([x]))

    return f1, fp, fpp, expr, grad_expr[0], hess_expr[0, 0]


def parse_vector(text: str, n: int | None = None):
    vals = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    if n is not None and len(vals) != n:
        raise ValueError(f"El vector debe tener {n} valores. Se recibieron {len(vals)}.")
    return np.array(vals, dtype=float)


def parse_matrix(text: str):
    rows = []
    for line in text.strip().splitlines():
        if line.strip():
            rows.append([float(x.strip()) for x in line.split(",")])
    return np.array(rows, dtype=float)


def detectar_operador(linea: str):
    for op in ["<=", ">=", "="]:
        if op in linea:
            return op
    raise ValueError(f"No se encontro operador <=, >= o = en la restriccion: {linea}")


def parsear_restricciones(text: str, n: int):
    """
    Acepta dos formatos:
    1) Simbolico: x1 + 2*x2 <= 30
    2) Por coeficientes: 1,2,30,<=
    Devuelve una lista de restricciones simbolicas en forma expr <= 0, expr >= 0 o expr == 0.
    """
    restricciones = []
    symbols = simbolos_n(n)

    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue

        # Formato antiguo por comas: a1,a2,...,b,<=
        if "," in line and not any(v in line for v in ["x1", "x2", "x3"]):
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != n + 2:
                raise ValueError(f"Restriccion por coeficientes invalida: {line}")
            coeffs = np.array([float(p) for p in parts[:n]], dtype=float)
            val = float(parts[n])
            op = parts[n + 1]
            expr = sum(coeffs[i] * symbols[i] for i in range(n)) - val
        else:
            op = detectar_operador(line)
            lhs, rhs = line.split(op, 1)
            expr = parsear_expresion(lhs, n) - parsear_expresion(rhs, n)

        if op not in ["<=", ">=", "="]:
            raise ValueError(f"Operador no soportado: {op}. Use <=, >= o =.")
        restricciones.append({"expr": sp.expand(expr), "op": op, "texto": line})

    return restricciones


def restricciones_scipy(restricciones, n: int):
    symbols = simbolos_n(n)
    cons = []
    for r in restricciones:
        expr = r["expr"]
        lamb = sp.lambdify(symbols, expr, "numpy")

        def fun(x, lamb=lamb, op=r["op"]):
            val = float(lamb(*np.array(x, dtype=float)))
            if op == "<=":
                return -val   # expr <= 0  => -expr >= 0
            if op == ">=":
                return val    # expr >= 0
            return val        # expr == 0

        cons.append({"type": "eq" if r["op"] == "=" else "ineq", "fun": fun})
    return cons


def restricciones_a_A_b(restricciones, n: int):
    """Convierte restricciones lineales a A x <= b. Las igualdades se duplican."""
    symbols = simbolos_n(n)
    A, b = [], []
    for r in restricciones:
        expr = sp.expand(r["expr"])
        # expr = a*x - b <= 0
        coeffs = [float(expr.coeff(s)) for s in symbols]
        const = float(expr.subs({s: 0 for s in symbols}))
        residuo = sp.expand(expr - sum(expr.coeff(s) * s for s in symbols) - const)
        if residuo != 0:
            raise ValueError("Las restricciones deben ser lineales para este metodo.")

        # coeffs*x + const <= 0  => coeffs*x <= -const
        if r["op"] == "<=":
            A.append(coeffs)
            b.append(-const)
        elif r["op"] == ">=":
            A.append([-c for c in coeffs])
            b.append(const)
        else:  # igualdad
            A.append(coeffs)
            b.append(-const)
            A.append([-c for c in coeffs])
            b.append(const)
    return np.array(A, dtype=float), np.array(b, dtype=float)


def evaluar_factibilidad(x, restricciones, n: int, tol: float = 1e-7):
    symbols = simbolos_n(n)
    vals = []
    for r in restricciones:
        lamb = sp.lambdify(symbols, r["expr"], "numpy")
        val = float(lamb(*np.array(x, dtype=float)))
        if r["op"] == "<=":
            ok = val <= tol
        elif r["op"] == ">=":
            ok = val >= -tol
        else:
            ok = abs(val) <= tol
        vals.append(ok)
    return all(vals)


def formato_vector(x):
    return "(" + ", ".join([f"{v:.6f}" for v in np.array(x, dtype=float)]) + ")"


def mostrar_metricas(x_opt, f_opt, clasificacion=""):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Punto optimo aproximado</div>
            <div class="metric-value">{formato_vector(x_opt)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Valor de la funcion</div>
            <div class="metric-value">{f_opt:.8f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Clasificacion</div>
            <div class="metric-value">{clasificacion or 'Calculado'}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ALGORITMOS
# ============================================================
def biseccion_1d(f_prime, a, b, tol, max_iter):
    hist = []
    fa, fb = f_prime(a), f_prime(b)
    if fa * fb > 0:
        raise ValueError("El intervalo no encierra una raiz de f'(x). Pruebe con otro [a,b].")

    c = a
    for i in range(int(max_iter)):
        c = (a + b) / 2
        fpc = f_prime(c)
        hist.append({"Iteracion": i + 1, "a": a, "b": b, "x": c, "f'(x)": fpc})
        if abs(fpc) < tol or abs(b - a) / 2 < tol:
            break
        if f_prime(a) * fpc < 0:
            b = c
        else:
            a = c
    return c, hist


def newton_1d(f_prime, f_double_prime, x0, tol, max_iter):
    hist = []
    x = float(x0)
    for i in range(int(max_iter)):
        fp = f_prime(x)
        fpp = f_double_prime(x)
        hist.append({"Iteracion": i + 1, "x": x, "f'(x)": fp, "f''(x)": fpp})
        if abs(fp) < tol:
            break
        if abs(fpp) < 1e-12:
            raise ValueError("La segunda derivada es cercana a cero. Newton no puede continuar.")
        x = x - fp / fpp
    return x, hist


def gradiente_nd(f, grad_f, x0, alpha, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = np.array(x0, dtype=float)
    signo = -1 if tipo == "Minimizar" else 1
    for i in range(int(max_iter)):
        g = grad_f(x)
        row = {"Iteracion": i + 1, "f(x)": f(x), "||grad||": np.linalg.norm(g)}
        for j, val in enumerate(x):
            row[f"x{j+1}"] = val
        hist.append(row)
        if np.linalg.norm(g) < tol:
            break
        x = x + signo * alpha * g
    return x, hist


def newton_nd(f, grad_f, hess_f, x0, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = np.array(x0, dtype=float)
    factor = 1 if tipo == "Minimizar" else -1

    for i in range(int(max_iter)):
        g_original = grad_f(x)
        H_original = hess_f(x)
        g = factor * g_original
        H = factor * H_original
        row = {"Iteracion": i + 1, "f(x)": f(x), "||grad||": np.linalg.norm(g_original)}
        for j, val in enumerate(x):
            row[f"x{j+1}"] = val
        hist.append(row)
        if np.linalg.norm(g_original) < tol:
            break
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -np.linalg.pinv(H) @ g

        # Control simple de paso para evitar saltos enormes
        paso = 1.0
        valor_actual = factor * f(x)
        aceptado = False
        for _ in range(20):
            x_trial = x + paso * p
            if factor * f(x_trial) <= valor_actual + 1e-4 * paso * np.dot(g, p):
                aceptado = True
                break
            paso *= 0.5
        x = x_trial if aceptado else x + p
    return x, hist


def vertices_factibles(f, restricciones, n, tipo="Minimizar"):
    A, b = restricciones_a_A_b(restricciones, n)
    m = A.shape[0]
    vertices = []

    if m < n:
        return None, []

    for indices in itertools.combinations(range(m), n):
        A_sub = A[list(indices), :]
        b_sub = b[list(indices)]
        if np.linalg.matrix_rank(A_sub) == n:
            x = np.linalg.solve(A_sub, b_sub)
            if np.all(A @ x <= b + 1e-7):
                vertices.append(x)

    unique = []
    for v in vertices:
        if not any(np.linalg.norm(v - u) < 1e-6 for u in unique):
            unique.append(v)

    hist = []
    for i, v in enumerate(unique):
        row = {"Candidato": i + 1, "f(x)": f(v)}
        for j, val in enumerate(v):
            row[f"x{j+1}"] = val
        hist.append(row)

    if not hist:
        return None, []
    opt = min(hist, key=lambda d: d["f(x)"]) if tipo == "Minimizar" else max(hist, key=lambda d: d["f(x)"])
    return np.array([opt[f"x{j+1}"] for j in range(n)], dtype=float), hist


def proyectar(y, restricciones, n):
    cons = restricciones_scipy(restricciones, n)
    res = minimize(lambda x: np.sum((x - y) ** 2), x0=np.array(y, dtype=float), constraints=cons, method="SLSQP")
    if not res.success:
        return np.array(y, dtype=float)
    return res.x


def gradiente_proyectado(f, grad_f, restricciones, n, x0, alpha, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = proyectar(np.array(x0, dtype=float), restricciones, n)
    signo = -1 if tipo == "Minimizar" else 1
    for i in range(int(max_iter)):
        g = grad_f(x)
        row = {"Iteracion": i + 1, "f(x)": f(x), "||grad||": np.linalg.norm(g)}
        for j, val in enumerate(x):
            row[f"x{j+1}"] = val
        hist.append(row)
        y = x + signo * alpha * g
        x_new = proyectar(y, restricciones, n)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, hist


def resolver_general_scipy(f, grad_f, restricciones, n, x0, tipo="Minimizar"):
    factor = 1 if tipo == "Minimizar" else -1
    cons = restricciones_scipy(restricciones, n)
    res = minimize(
        lambda x: factor * f(x),
        x0=np.array(x0, dtype=float),
        jac=lambda x: factor * grad_f(x),
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    return res


def extraer_modelo_cuadratico(expr, n):
    symbols = simbolos_n(n)
    H = sp.Matrix([[sp.diff(expr, s1, s2) for s2 in symbols] for s1 in symbols])
    grad0 = sp.Matrix([sp.diff(expr, s).subs({v: 0 for v in symbols}) for s in symbols])
    const = expr.subs({v: 0 for v in symbols})

    # Forma general: f(x) = const + c^T x + 1/2 x^T H x
    # Forma de las diapositivas para maximizar: f(x)=c^T x - 1/2 x^T Q x => Q = -H
    Q_diapo = -H
    return H, grad0, const, Q_diapo

# ============================================================
# GRAFICAS MEJORADAS
# ============================================================
def layout_plotly(fig, titulo, x_title="x1", y_title="x2"):
    fig.update_layout(
        title={"text": titulo, "x": 0.03, "xanchor": "left"},
        template="plotly_white",
        height=640,
        margin=dict(l=30, r=30, t=70, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
        xaxis=dict(title=x_title, showgrid=True, zeroline=True, zerolinewidth=2),
        yaxis=dict(title=y_title, showgrid=True, zeroline=True, zerolinewidth=2, scaleanchor="x", scaleratio=1),
    )
    return fig


def graficar_1d(f, x_opt, a, b, hist=None):
    lo = min(a, b, x_opt) - 2
    hi = max(a, b, x_opt) + 2
    xs = np.linspace(lo, hi, 700)
    ys = np.array([f(x) for x in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(width=4, color="#0F4C81")))
    fig.add_trace(go.Scatter(x=[x_opt], y=[f(x_opt)], mode="markers", name="Punto critico", marker=dict(size=15, color="#E63946", symbol="star")))

    if hist:
        hx = [row["x"] for row in hist if "x" in row]
        if hx:
            hy = [f(v) for v in hx]
            fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers+lines", name="Iteraciones", marker=dict(size=7, color="#2A9D8F"), line=dict(width=2, dash="dot")))

    fig.update_layout(
        title="Funcion en una variable",
        template="plotly_white",
        height=580,
        xaxis_title="x",
        yaxis_title="f(x)",
        margin=dict(l=30, r=30, t=70, b=35),
    )
    return fig


def puntos_base_para_rango(x_opt, hist=None):
    pts = []
    if x_opt is not None:
        pts.append(np.array(x_opt[:2], dtype=float))
    if hist:
        for h in hist:
            if "x1" in h and "x2" in h:
                pts.append(np.array([h["x1"], h["x2"]], dtype=float))
    return pts


def rango_grafica_2d(x_opt, restricciones=None, hist=None):
    pts = puntos_base_para_rango(x_opt, hist)
    pts.append(np.array([0, 0], dtype=float))

    # Si hay restricciones lineales, tratar de incluir vertices factibles.
    if restricciones is not None:
        try:
            A, b = restricciones_a_A_b(restricciones, 2)
            for i, j in itertools.combinations(range(A.shape[0]), 2):
                M = A[[i, j], :]
                q = b[[i, j]]
                if np.linalg.matrix_rank(M) == 2:
                    p = np.linalg.solve(M, q)
                    if np.all(A @ p <= b + 1e-6):
                        pts.append(p)
        except Exception:
            pass

    arr = np.array(pts, dtype=float)
    xmin, ymin = np.min(arr, axis=0)
    xmax, ymax = np.max(arr, axis=0)
    if abs(xmax - xmin) < 2:
        xmin -= 5
        xmax += 5
    if abs(ymax - ymin) < 2:
        ymin -= 5
        ymax += 5
    dx = max(2.0, 0.25 * (xmax - xmin))
    dy = max(2.0, 0.25 * (ymax - ymin))
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def crear_mascara_factible(X, Y, restricciones):
    if not restricciones:
        return np.ones_like(X, dtype=bool)
    mask = np.ones_like(X, dtype=bool)
    x1, x2 = simbolos_n(2)
    for r in restricciones:
        lamb = sp.lambdify((x1, x2), r["expr"], "numpy")
        val = lamb(X, Y)
        if r["op"] == "<=":
            mask &= val <= 1e-7
        elif r["op"] == ">=":
            mask &= val >= -1e-7
        else:
            mask &= np.abs(val) <= 1e-3
    return mask


def graficar_2d(f, x_opt, hist=None, restricciones=None, titulo="Curvas de nivel y region factible"):
    xmin, xmax, ymin, ymax = rango_grafica_2d(x_opt, restricciones, hist)
    x_range = np.linspace(xmin, xmax, 260)
    y_range = np.linspace(ymin, ymax, 260)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f([X[i, j], Y[i, j]])
            except Exception:
                Z[i, j] = np.nan

    mask = crear_mascara_factible(X, Y, restricciones)
    Z_visible = np.where(mask, Z, np.nan)

    fig = go.Figure()

    if restricciones:
        fig.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=np.where(mask, 1, np.nan),
            showscale=False,
            colorscale=[[0, "rgba(46, 196, 182, 0.16)"], [1, "rgba(46, 196, 182, 0.16)"]],
            contours=dict(start=0, end=1, size=1, coloring="heatmap"),
            name="Region factible",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z_visible,
        colorscale="Viridis",
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=11, color="white")),
        colorbar=dict(title="f(x1,x2)"),
        opacity=0.88,
        name="Funcion objetivo",
    ))

    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z_visible,
        contours=dict(coloring="lines", showlabels=False),
        line=dict(width=1.5, color="rgba(255,255,255,0.65)"),
        showscale=False,
        name="Curvas de nivel",
        hoverinfo="skip",
    ))

    if restricciones:
        x1, x2 = simbolos_n(2)
        for idx, r in enumerate(restricciones, 1):
            expr = sp.expand(r["expr"])
            # Intentar graficar frontera expr=0
            try:
                sol_y = sp.solve(sp.Eq(expr, 0), x2)
                if sol_y:
                    y_vals = sp.lambdify(x1, sol_y[0], "numpy")(x_range)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_vals,
                        mode="lines",
                        name=f"Restriccion {idx}: {r['texto']}",
                        line=dict(width=3, color="#E76F51"),
                    ))
                    continue
                sol_x = sp.solve(sp.Eq(expr, 0), x1)
                if sol_x:
                    x_vals = np.ones_like(y_range) * float(sol_x[0])
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_range,
                        mode="lines",
                        name=f"Restriccion {idx}: {r['texto']}",
                        line=dict(width=3, color="#E76F51"),
                    ))
            except Exception:
                pass

    if hist:
        hx = [h["x1"] for h in hist if "x1" in h and "x2" in h]
        hy = [h["x2"] for h in hist if "x1" in h and "x2" in h]
        if hx:
            fig.add_trace(go.Scatter(
                x=hx,
                y=hy,
                mode="lines+markers",
                name="Iteraciones",
                marker=dict(size=8, color="#FFD166"),
                line=dict(width=3, color="#FFD166"),
            ))

    fig.add_trace(go.Scatter(
        x=[x_opt[0]],
        y=[x_opt[1]],
        mode="markers+text",
        text=["Optimo"],
        textposition="top center",
        name="Optimo",
        marker=dict(size=18, color="#D00000", symbol="star", line=dict(width=2, color="white")),
    ))

    return layout_plotly(fig, titulo)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## MENU")
opcion = st.sidebar.radio(
    "Seleccione el modulo:",
    [
        "1. Una variable",
        "2. Varias variables",
        "3. Restringida linealmente",
        "4. Programacion cuadratica",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(f"Modulo activo:\n\n{opcion}")
st.sidebar.markdown(
    """
    ### Recomendaciones
    - Use `*` para multiplicar: `4*x1*x2`.
    - Tambien puede usar `^` para potencias.
    - En restricciones use `<=`, `>=` o `=`.
    """
)

# ============================================================
# MODULO 1
# ============================================================
if opcion.startswith("1"):
    st.markdown("## 1. Optimizacion no restringida de una variable")
    col1, col2 = st.columns([0.92, 1.7], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Datos del problema")
        expr_str = st.text_input("Funcion f(x):", value="x^3 - 6*x^2 + 9*x + 1")
        metodo = st.selectbox("Metodo:", ["Biseccion", "Newton"])
        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"])
        c1, c2 = st.columns(2)
        a = c1.number_input("a:", value=0.0)
        b = c2.number_input("b:", value=3.0)
        x0 = st.number_input("x0 Newton:", value=1.5)
        c3, c4 = st.columns(2)
        tol = c3.number_input("Tolerancia:", value=0.0001, format="%.6f")
        max_iter = c4.number_input("Iteraciones:", value=50, min_value=1, step=1)
        resolver = st.button("Resolver")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if resolver:
            try:
                f, fp, fpp, expr, expr_p, expr_pp = crear_funcion_1d(expr_str)
                if metodo == "Biseccion":
                    x_opt, hist = biseccion_1d(fp, a, b, tol, max_iter)
                else:
                    x_opt, hist = newton_1d(fp, fpp, x0, tol, max_iter)
                f_opt = f(x_opt)
                segunda = fpp(x_opt)
                if segunda > 1e-8:
                    clasificacion = "Minimo local"
                elif segunda < -1e-8:
                    clasificacion = "Maximo local"
                else:
                    clasificacion = "No concluyente"

                mostrar_metricas([x_opt], f_opt, clasificacion)
                t1, t2, t3 = st.tabs(["Respuesta", "Iteraciones", "Grafica"])
                with t1:
                    st.latex(f"f(x) = {sp.latex(expr)}")
                    st.latex(f"f'(x) = {sp.latex(expr_p)}")
                    st.latex(f"f''(x) = {sp.latex(expr_pp)}")
                    st.write(f"**Objetivo seleccionado:** {tipo}")
                    st.write("Nota: el metodo encuentra puntos criticos; la segunda derivada determina si es minimo o maximo local.")
                with t2:
                    st.dataframe(pd.DataFrame(hist), use_container_width=True)
                with t3:
                    st.plotly_chart(graficar_1d(f, x_opt, a, b, hist), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# MODULO 2
# ============================================================
elif opcion.startswith("2"):
    st.markdown("## 2. Optimizacion no restringida de varias variables")
    col1, col2 = st.columns([0.92, 1.7], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Datos del problema")
        n = st.selectbox("Numero de variables:", [2, 3])
        expr_default = "x1^2 + x2^2 - 4*x1 - 6*x2 + 13" if n == 2 else "x1^2 + x2^2 + x3^2"
        expr_str = st.text_input("Funcion:", value=expr_default)
        metodo = st.selectbox("Metodo:", ["Gradiente", "Newton", "SciPy SLSQP"])
        tipo = st.selectbox("Objetivo:", ["Minimizar", "Maximizar"])
        p0_str = st.text_input("Punto inicial:", value="0,0" if n == 2 else "0,0,0")
        alpha = st.number_input("Alpha:", value=0.1)
        c1, c2 = st.columns(2)
        tol = c1.number_input("Tolerancia:", value=0.0001, format="%.6f")
        max_iter = c2.number_input("Iteraciones:", value=80, min_value=1, step=1)
        resolver = st.button("Resolver")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if resolver:
            try:
                x0 = parse_vector(p0_str, n)
                f, grad_f, hess_f, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, n)
                if metodo == "Gradiente":
                    x_opt, hist = gradiente_nd(f, grad_f, x0, alpha, tol, max_iter, tipo)
                elif metodo == "Newton":
                    x_opt, hist = newton_nd(f, grad_f, hess_f, x0, tol, max_iter, tipo)
                else:
                    res = resolver_general_scipy(f, grad_f, [], n, x0, tipo)
                    x_opt, hist = res.x, []

                f_opt = f(x_opt)
                eigvals = np.linalg.eigvals(hess_f(x_opt))
                if np.all(eigvals > 1e-8):
                    clasificacion = "Minimo local"
                elif np.all(eigvals < -1e-8):
                    clasificacion = "Maximo local"
                elif np.any(eigvals > 1e-8) and np.any(eigvals < -1e-8):
                    clasificacion = "Punto silla"
                else:
                    clasificacion = "No concluyente"

                mostrar_metricas(x_opt, f_opt, clasificacion)
                t1, t2, t3 = st.tabs(["Respuesta", "Iteraciones", "Grafica"])
                with t1:
                    st.latex(f"f(x) = {sp.latex(expr)}")
                    st.write("**Gradiente:**")
                    st.latex(sp.latex(sp.Matrix(grad_expr)))
                    st.write("**Hessiana:**")
                    st.latex(sp.latex(hess_expr))
                    st.write(f"**Autovalores Hessiana:** {np.round(eigvals, 6)}")
                with t2:
                    if hist:
                        st.dataframe(pd.DataFrame(hist), use_container_width=True)
                    else:
                        st.info("El metodo SciPy devuelve el optimo directamente, sin tabla de iteraciones detallada.")
                with t3:
                    if n == 2:
                        st.plotly_chart(graficar_2d(f, x_opt, hist, None, "Curvas de nivel"), use_container_width=True)
                    else:
                        st.info("La grafica de curvas de nivel esta disponible para 2 variables.")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# MODULO 3
# ============================================================
elif opcion.startswith("3"):
    st.markdown("## 3. Optimizacion restringida linealmente")
    col1, col2 = st.columns([0.92, 1.7], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Datos del problema")
        n = st.selectbox("Numero de variables:", [2, 3])
        expr_str = st.text_input("Funcion objetivo:", value="x1*x2 + 2*x1 + x2")
        metodo = st.selectbox("Metodo:", ["Gradiente proyectado", "SciPy SLSQP", "Vertices factibles"])
        tipo = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])
        restricciones_str = st.text_area(
            "Restricciones:",
            value="x1 <= 4\nx2 <= 6\n3*x1 + 2*x2 <= 18\nx1 >= 0\nx2 >= 0",
            height=140,
        )
        p0_str = st.text_input("Punto inicial:", value="1,1" if n == 2 else "1,1,1")
        alpha = st.number_input("Alpha:", value=0.15)
        c1, c2 = st.columns(2)
        tol = c1.number_input("Tol:", value=0.0001, format="%.6f")
        max_iter = c2.number_input("Iteraciones:", value=100, min_value=1, step=1)
        resolver = st.button("Resolver")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if resolver:
            try:
                x0 = parse_vector(p0_str, n)
                restricciones = parsear_restricciones(restricciones_str, n)
                f, grad_f, hess_f, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, n)

                if metodo == "Vertices factibles":
                    x_opt, hist = vertices_factibles(f, restricciones, n, tipo)
                    if x_opt is None:
                        raise ValueError("No se encontraron vertices factibles. Revise las restricciones.")
                elif metodo == "Gradiente proyectado":
                    x_opt, hist = gradiente_proyectado(f, grad_f, restricciones, n, x0, alpha, tol, max_iter, tipo)
                else:
                    res = resolver_general_scipy(f, grad_f, restricciones, n, x0, tipo)
                    if not res.success:
                        st.warning(f"SciPy aviso: {res.message}")
                    x_opt, hist = res.x, []

                f_opt = f(x_opt)
                mostrar_metricas(x_opt, f_opt, "Factible" if evaluar_factibilidad(x_opt, restricciones, n) else "Revisar")
                t1, t2, t3 = st.tabs(["Respuesta", "Tabla", "Grafica"])
                with t1:
                    st.latex(f"f(x) = {sp.latex(expr)}")
                    st.write("**Restricciones ingresadas:**")
                    for r in restricciones:
                        st.code(r["texto"])
                    st.write(f"**Metodo:** {metodo}")
                    if metodo == "Vertices factibles":
                        st.warning("Para funciones no lineales, revisar solo vertices no garantiza siempre el optimo global. Se recomienda SciPy SLSQP o gradiente proyectado.")
                with t2:
                    if hist:
                        st.dataframe(pd.DataFrame(hist), use_container_width=True)
                    else:
                        st.info("Este metodo no genera tabla iterativa detallada.")
                with t3:
                    if n == 2:
                        st.plotly_chart(graficar_2d(f, x_opt, hist, restricciones, "Funcion objetivo con region factible"), use_container_width=True)
                    else:
                        st.info("La grafica esta disponible para 2 variables.")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# MODULO 4 - CUADRATICA DIRECTA DESDE LA FUNCION
# ============================================================
elif opcion.startswith("4"):
    st.markdown("## 4. Programacion cuadratica")
    st.markdown(
        """
        <div class="info-card">
        En este modulo ya no es necesario ingresar Q, c y constante por separado. Escriba directamente la funcion completa y las restricciones.
        El programa identifica la Hessiana, el vector lineal y la matriz Q usada en la forma de las diapositivas.
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([0.92, 1.7], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Datos del problema cuadratico")
        tipo = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])
        expr_str = st.text_area(
            "Funcion objetivo f(x1,x2):",
            value="15*x1 + 30*x2 + 4*x1*x2 - 2*x1^2 - 4*x2^2",
            height=90,
        )
        restricciones_str = st.text_area(
            "Restricciones:",
            value="x1 + 2*x2 <= 30\nx1 >= 0\nx2 >= 0",
            height=120,
        )
        p0_str = st.text_input("Punto inicial:", value="1,1")
        resolver = st.button("Resolver cuadratica")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if resolver:
            try:
                n = 2
                x0 = parse_vector(p0_str, n)
                restricciones = parsear_restricciones(restricciones_str, n)
                f, grad_f, hess_f, expr, grad_expr, hess_expr = crear_funcion_nd(expr_str, n)
                H, c_vec, const, Q_diapo = extraer_modelo_cuadratico(expr, n)

                # Verificacion: una funcion cuadratica tiene Hessiana constante.
                x1, x2 = simbolos_n(2)
                if any(term.has(x1, x2) for term in list(H)):
                    st.warning("La funcion parece no ser cuadratica pura, pero se intentara resolver numericamente.")

                res = resolver_general_scipy(f, grad_f, restricciones, n, x0, tipo)
                if not res.success:
                    st.warning(f"SciPy aviso: {res.message}")
                x_opt = res.x
                f_opt = f(x_opt)

                H_num = np.array(H, dtype=float)
                eig_H = np.linalg.eigvals(H_num)
                if tipo == "Maximizar" and np.all(eig_H < -1e-8):
                    clasificacion = "Concava: maximo confiable"
                elif tipo == "Minimizar" and np.all(eig_H > 1e-8):
                    clasificacion = "Convexa: minimo confiable"
                elif np.any(eig_H > 1e-8) and np.any(eig_H < -1e-8):
                    clasificacion = "Indefinida"
                else:
                    clasificacion = "Semidefinida / revisar"

                mostrar_metricas(x_opt, f_opt, clasificacion)
                t1, t2, t3, t4 = st.tabs(["Respuesta", "Modelo matricial", "Restricciones", "Grafica"])

                with t1:
                    st.latex(f"f(x_1,x_2) = {sp.latex(expr)}")
                    st.write(f"**Objetivo:** {tipo}")
                    st.write(f"**x1 optimo:** {x_opt[0]:.8f}")
                    st.write(f"**x2 optimo:** {x_opt[1]:.8f}")
                    st.write(f"**f optimo:** {f_opt:.8f}")
                    st.write(f"**Factibilidad:** {'Cumple restricciones' if evaluar_factibilidad(x_opt, restricciones, n) else 'No cumple, revisar'}")

                with t2:
                    st.write("**Forma general:**")
                    st.latex(r"f(x)=k+c^Tx+\frac{1}{2}x^T Hx")
                    st.write("**Forma usada en las diapositivas para maximizar:**")
                    st.latex(r"f(x)=c^Tx-\frac{1}{2}x^TQx")
                    st.write("**Vector c:**")
                    st.latex(sp.latex(c_vec.T))
                    st.write("**Hessiana H:**")
                    st.latex(sp.latex(H))
                    st.write("**Matriz Q = -H:**")
                    st.latex(sp.latex(Q_diapo))
                    st.write(f"**Constante k:** {float(const):.6f}")
                    st.write(f"**Autovalores de H:** {np.round(eig_H, 6)}")

                with t3:
                    st.write("**Restricciones ingresadas:**")
                    for r in restricciones:
                        st.code(r["texto"])
                    try:
                        A, b = restricciones_a_A_b(restricciones, n)
                        st.write("**Forma A x <= b:**")
                        st.write("A =")
                        st.dataframe(pd.DataFrame(A, columns=["x1", "x2"]), use_container_width=True)
                        st.write("b =")
                        st.dataframe(pd.DataFrame(b, columns=["b"]), use_container_width=True)
                    except Exception:
                        st.info("No todas las restricciones se pudieron convertir a A x <= b.")

                with t4:
                    st.plotly_chart(graficar_2d(f, x_opt, [], restricciones, "Programacion cuadratica: curvas de nivel y region factible"), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

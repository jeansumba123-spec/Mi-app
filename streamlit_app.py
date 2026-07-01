import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(
    page_title="Examen Final - Investigación de Operaciones",
    layout="wide"
)

# ==============================
# ENCABEZADO PRINCIPAL
# ==============================
st.title("📘 Examen Final de Investigación de Operaciones")
st.subheader("Programación No Lineal")

st.markdown("---")

# ==============================
# INTEGRANTES
# ==============================
st.sidebar.title("👨‍🎓 Integrantes")
st.sidebar.write("• Jean Sumba")
st.sidebar.write("• Juan Pacheco")

st.sidebar.markdown("---")

# ==============================
# MENÚ PRINCIPAL
# ==============================
menu = st.sidebar.selectbox(
    "📌 Seleccione el tema",
    [
        "Inicio",
        "1. Optimización no restringida (1 variable)",
        "2. Optimización no restringida (varias variables)",
        "3. Optimización restringida linealmente",
        "4. Optimización cuadrática"
    ]
)

# ==============================
# PÁGINA INICIO
# ==============================
if menu == "Inicio":
    st.write("## Bienvenido al sistema")
    st.write("""
Este proyecto corresponde al **examen final de Investigación de Operaciones**.

Permite resolver problemas de **programación no lineal** basados en métodos vistos en clase y en el documento proporcionado:

- Método de Bisección  
- Método de Newton  
- Gradiente  
- Optimización cuadrática  

Selecciona una opción en el menú para comenzar.
    """)

# ==============================
# 1 VARIABLE
# ==============================

elif menu == "1. Optimización no restringida (1 variable)":

    st.header("🔹 Optimización no restringida de una sola variable")
    st.subheader("📌 Método de Bisección (basado en f'(x)=0)")

    x = sp.Symbol('x')

    # =========================
    # INPUTS
    # =========================
    func_input = st.text_input("Ingresa f(x)", "x^3 - 6*x^2 + 9*x + 1")

    a = st.number_input("Intervalo a", value=0.0)
    b = st.number_input("Intervalo b", value=3.0)

    tol = st.number_input("Tolerancia", value=0.0001, format="%.6f")
    max_iter = st.number_input("Iteraciones", value=50)

    # =========================
    # BOTÓN
    # =========================
    if st.button("🚀 Ejecutar Bisección"):

        try:
            f_sym = sp.sympify(func_input)
            df_sym = sp.diff(f_sym, x)

            f = sp.lambdify(x, f_sym, "numpy")
            df = sp.lambdify(x, df_sym, "numpy")

            st.latex(f"f(x) = {sp.latex(f_sym)}")
            st.latex(f"f'(x) = {sp.latex(df_sym)}")

            if df(a) * df(b) > 0:
                st.error("❌ f'(a)*f'(b) debe ser < 0. Cambia el intervalo.")
            else:

                iter_list = []
                m_prev = 0

                for i in range(int(max_iter)):

                    m = (a + b) / 2

                    fa = df(a)
                    fm = df(m)

                    error = abs(m - m_prev) if i > 0 else abs(b - a)

                    iter_list.append([i, a, b, m, fa, fm, error])

                    if abs(fm) < tol or error < tol:
                        break

                    if fa * fm < 0:
                        b = m
                    else:
                        a = m

                    m_prev = m

                import pandas as pd
                df_table = pd.DataFrame(iter_list, columns=[
                    "Iter", "a", "b", "m", "f'(a)", "f'(m)", "Error"
                ])

                st.subheader("📊 Iteraciones")
                st.dataframe(df_table)

                x_opt = df_table["m"].iloc[-1]
                y_opt = f(x_opt)

                st.success(f"✔ Óptimo: x = {x_opt}")
                st.success(f"✔ f(x) = {y_opt}")

                import matplotlib.pyplot as plt
                import numpy as np

                xs = np.linspace(x_opt - 3, x_opt + 3, 300)
                ys = f(xs)

                plt.figure()
                plt.plot(xs, ys, label="f(x)")
                plt.axvline(x_opt, color="red", linestyle="--")
                plt.scatter([x_opt], [y_opt], color="red")
                plt.grid()
                plt.legend()

                st.pyplot(plt)

        except Exception as e:
            st.error(f"Error: {e}")
elif menu == "2. Optimización no restringida (varias variables)":
    st.header("🔹 Optimización no restringida de varias variables")
    st.write("Aquí se implementarán métodos de gradiente y Newton multivariable.")

# ==============================
# RESTRICCIÓN LINEAL
# ==============================
elif menu == "3. Optimización restringida linealmente":
    st.header("🔹 Optimización con restricciones lineales")
    st.write("Aquí se aplicará programación lineal (método gráfico o simplex).")

# ==============================
# CUADRÁTICA
# ==============================
elif menu == "4. Optimización cuadrática":
    st.header("🔹 Optimización cuadrática")
    st.write("Aquí se resolverán funciones cuadráticas con métodos numéricos o analíticos.")



import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.header("🔹 Optimización no restringida (1 variable)")
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
        # =========================
        # FUNCIONES
        # =========================
        f_sym = sp.sympify(func_input)
        df_sym = sp.diff(f_sym, x)

        f = sp.lambdify(x, f_sym, "numpy")
        df = sp.lambdify(x, df_sym, "numpy")

        st.latex(f"f(x) = {sp.latex(f_sym)}")
        st.latex(f"f'(x) = {sp.latex(df_sym)}")

        # =========================
        # VALIDACIÓN (como MATLAB)
        # =========================
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

            # =========================
            # TABLA
            # =========================
            df_table = pd.DataFrame(iter_list, columns=[
                "Iter", "a", "b", "m", "f'(a)", "f'(m)", "Error"
            ])

            st.subheader("📊 Iteraciones")
            st.dataframe(df_table)

            x_opt = df_table["m"].iloc[-1]
            y_opt = f(x_opt)

            st.success(f"✔ Óptimo aproximado: x = {x_opt}")
            st.success(f"✔ f(x) = {y_opt}")

            # =========================
            # GRÁFICA
            # =========================
            st.subheader("📈 Gráfica")

            xs = np.linspace(x_opt - 3, x_opt + 3, 300)
            ys = f(xs)

            plt.figure()
            plt.plot(xs, ys, label="f(x)")
            plt.axvline(x_opt, color="red", linestyle="--", label="Óptimo")
            plt.scatter([x_opt], [y_opt], color="red")

            plt.grid()
            plt.legend()

            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {e}")

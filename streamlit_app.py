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
            # =========================
            # FUNCIONES
            # =========================
            f_sym = sp.sympify(func_input)
            df_sym = sp.diff(f_sym, x)

            f = sp.lambdify(x, f_sym, "numpy")
            df = sp.lambdify(x, df_sym, "numpy")

            # =========================
            # TEXTO TIPO INFORME (como MATLAB)
            # =========================
            st.markdown("## 📄 Análisis del problema")

            st.write("Se busca encontrar un punto crítico resolviendo:")
            st.latex("f'(x) = 0")

            st.write("Función original:")
            st.latex(f"f(x) = {sp.latex(f_sym)}")

            st.write("Derivada:")
            st.latex(f"f'(x) = {sp.latex(df_sym)}")

            # =========================
            # VALIDACIÓN
            # =========================
            if df(a) * df(b) > 0:
                st.error("❌ El método no es válido: f'(a)*f'(b) debe ser negativo.")
            else:

                iter_list = []
                m_prev = None

                st.markdown("## 🔁 Proceso iterativo")

                for i in range(int(max_iter)):

                    m = (a + b) / 2

                    fa = df(a)
                    fm = df(m)

                    error = abs(m - m_prev) if m_prev is not None else abs(b - a)

                    iter_list.append([
                        i, a, b, m, fa, fm, error
                    ])

                    st.write(
                        f"Iteración {i}: a={a:.4f}, b={b:.4f}, m={m:.4f}, f'(m)={fm:.4f}, error={error:.6f}"
                    )

                    if abs(fm) < tol or error < tol:
                        break

                    if fa * fm < 0:
                        b = m
                    else:
                        a = m

                    m_prev = m

                # =========================
                # TABLA FINAL (tipo libro)
                # =========================
                df_table = pd.DataFrame(iter_list, columns=[
                    "Iteración", "a", "b", "x_medio", "f'(a)", "f'(m)", "Error"
                ])

                st.markdown("## 📊 Tabla de iteraciones (Método de Bisección)")
                st.dataframe(df_table, use_container_width=True)

                # =========================
                # RESULTADO FINAL
                # =========================
                x_opt = df_table["x_medio"].iloc[-1]
                y_opt = f(x_opt)

                st.markdown("## 🎯 Resultado final")

                st.success(f"✔ Punto óptimo aproximado: x = {x_opt}")
                st.success(f"✔ f(x) = {y_opt}")

                # =========================
                # INTERPRETACIÓN (como libro / MATLAB)
                # =========================
                st.markdown("## 🧠 Interpretación")

                st.write("""
El método de bisección aplicado a la derivada permite aproximar el punto donde:

- f'(x) ≈ 0  
- Esto indica un punto crítico de la función  
- Dependiendo de la segunda derivada, puede ser mínimo o máximo local  
                """)

                # =========================
                # GRÁFICA
                # =========================
                st.markdown("## 📈 Gráfica de la función")

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



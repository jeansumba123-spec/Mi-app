import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIGURACIÓN
# ==============================
st.set_page_config(
    page_title="Examen Final - IO",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
st.title("📘 Examen Final de Investigación de Operaciones")
st.subheader("Programación No Lineal")

st.sidebar.title("👨‍🎓 Integrantes")
st.sidebar.write("Jean Sumba")
st.sidebar.write("Juan Pacheco")

menu = st.sidebar.selectbox(
    "Seleccione el tema",
    [
        "Inicio",
        "1. Bisección (1 variable)",
        "2. Varias variables",
        "3. Linealmente restringida",
        "4. Cuadrática"
    ]
)

# ==============================
# FUNCIONES DEL EJEMPLO (DIAPOSITIVAS)
# ==============================
def f(x):
    return 12*x - 3*x**4 - 2*x**6

def df(x):
    return 12*(1 - x**3 - x**5)

# ==============================
# BISSECTION METHOD (como diapositivas)
# ==============================
def biseccion(a, b, tol=1e-4, max_iter=50):
    data = []

    for i in range(max_iter):
        x = (a + b) / 2
        fx = df(x)

        data.append([i, a, b, x, fx])

        if abs(fx) < tol:
            break

        if df(a) * fx < 0:
            b = x
        else:
            a = x

    return pd.DataFrame(data, columns=["Iteración", "a", "b", "x*", "df(x*)"])

# ==============================
# NAVEGACIÓN
# ==============================
if menu == "Inicio":
    st.write("Sistema de optimización no lineal basado en el PDF de clase.")

elif menu == "1. Bisección (1 variable)":

    st.header("Método de Bisección - Optimización no restringida")

    st.image("/mnt/data/image(1120).png", caption="Ejemplo del método en diapositivas")

    st.latex(r"f(x)=12x - 3x^4 - 2x^6")
    st.latex(r"f'(x)=12(1 - x^3 - x^5)")

    st.write("### Parámetros del método")
    a = st.number_input("Valor inicial a", value=0.0)
    b = st.number_input("Valor inicial b", value=2.0)
    tol = st.number_input("Tolerancia", value=0.0001, format="%.5f")

    if st.button("Ejecutar Bisección"):

        df_result = biseccion(a, b, tol)

        st.write("### Tabla de iteraciones")
        st.dataframe(df_result)

        st.write("### Resultado aproximado")
        st.success(f"x* ≈ {df_result.iloc[-1, 3]}")

        # gráfico
        xs = np.linspace(a, b, 200)
        ys = f(xs)

        plt.figure()
        plt.plot(xs, ys)
        plt.axvline(df_result.iloc[-1, 3], color="red")
        plt.title("Función objetivo")
        st.pyplot(plt)

elif menu == "2. Varias variables":
    st.header("Pendiente implementación")

elif menu == "3. Linealmente restringida":
    st.header("Pendiente implementación")

elif menu == "4. Cuadrática":
    st.header("Pendiente implementación")

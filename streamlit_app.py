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
    st.write("Aquí se implementarán métodos como Bisección y Newton para una variable.")

# ==============================
# VARIAS VARIABLES
# ==============================
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

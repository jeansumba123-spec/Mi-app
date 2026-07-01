import streamlit as st

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(
    page_title="Optimización No Lineal",
    layout="wide"
)

# ==============================
# TÍTULO PRINCIPAL
# ==============================
st.title("📊 Sistema de Optimización Matemática")
st.subheader("Programación No Lineal - Investigación de Operaciones")

st.markdown("---")

# ==============================
# MENÚ PRINCIPAL
# ==============================
menu = st.sidebar.selectbox(
    "📌 Seleccione el tipo de optimización",
    [
        "Inicio",
        "1. Optimización no restringida (1 variable)",
        "2. Optimización no restringida (varias variables)",
        "3. Optimización restringida linealmente",
        "4. Optimización cuadrática"
    ]
)

# ==============================
# INICIO
# ==============================
if menu == "Inicio":
    st.write("## Bienvenido")
    st.write("""
    Este sistema permite resolver problemas de programación no lineal basado en métodos numéricos:

    - Método de Bisección  
    - Método de Newton  
    - Gradiente  
    - Optimización cuadrática  

    Selecciona una opción en el menú lateral para comenzar.
    """)

# ==============================
# 1 VARIABLE
# ==============================
elif menu == "1. Optimización no restringida (1 variable)":
    st.header("🔹 Optimización no restringida de una sola variable")
    st.write("Aquí luego implementaremos Bisección y Newton para una variable.")

# ==============================
# VARIAS VARIABLES
# ==============================
elif menu == "2. Optimización no restringida (varias variables)":
    st.header("🔹 Optimización no restringida de varias variables")
    st.write("Aquí implementaremos gradiente y Newton multivariable.")

# ==============================
# RESTRINGIDA LINEAL
# ==============================
elif menu == "3. Optimización restringida linealmente":
    st.header("🔹 Optimización con restricciones lineales")
    st.write("Aquí se implementará programación lineal (posiblemente método gráfico o simplex).")

# ==============================
# CUADRÁTICA
# ==============================
elif menu == "4. Optimización cuadrática":
    st.header("🔹 Optimización cuadrática")
    st.write("Aquí se resolverán funciones cuadráticas con métodos analíticos o numéricos.")

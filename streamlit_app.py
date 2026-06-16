import streamlit as st
import math
import copy
import numpy as np
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Optimizador IP - Branch & Bound", layout="wide")

st.title("Optimizador de Programación Lineal Entera")
st.subheader("Branch & Bound con n variables y n restricciones")

st.markdown("""
Esta aplicación permite resolver problemas de programación lineal entera usando el método **Branch & Bound**.

También incluye el ejercicio de **Euing Gas**, donde se desea maximizar la ganancia al producir dos tipos de gasolina a partir de dos aceites.
""")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def es_entero(valor, tol=1e-6):
    return abs(valor - round(valor)) <= tol


def mejor_que(z1, z2, es_max):
    if es_max:
        return z1 > z2 + 1e-6
    return z1 < z2 - 1e-6


def peor_o_igual_que(z1, z2, es_max):
    if es_max:
        return z1 <= z2 + 1e-6
    return z1 >= z2 - 1e-6


def inicializar_euing_gas():
    """
    Variables:
    x1 = G1  = galones de gasolina 1 producidos
    x2 = G2  = galones de gasolina 2 producidos
    x3 = A11 = aceite 1 usado en gas 1
    x4 = A12 = aceite 2 usado en gas 1
    x5 = A21 = aceite 1 usado en gas 2
    x6 = A22 = aceite 2 usado en gas 2
    x7 = P1  = compra de aceite 1 en el primer bloque, máximo 500 galones a 25 centavos
    x8 = P2  = compra de aceite 1 en el segundo bloque, máximo 500 galones a 20 centavos
    x9 = P3  = compra de aceite 1 en el tercer bloque, máximo 500 galones a 15 centavos
    x10 = Y2 = variable binaria para activar segundo bloque
    x11 = Y3 = variable binaria para activar tercer bloque
    """

    st.session_state.tipo_opt = "Maximizar"
    st.session_state.n_vars = 11
    st.session_state.n_rest = 12

    nombres = [
        "G1",
        "G2",
        "A11",
        "A12",
        "A21",
        "A22",
        "P1",
        "P2",
        "P3",
        "Y2",
        "Y3"
    ]

    tipos = [
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Entera",
        "Binaria",
        "Binaria"
    ]

    # Función objetivo en centavos:
    # Max Z = 12G1 + 14G2 - 25P1 - 20P2 - 15P3
    c = [12, 14, 0, 0, 0, 0, -25, -20, -15, 0, 0]

    lb = [0] * 11
    ub = [
        None,
        None,
        None,
        None,
        None,
        None,
        500,
        500,
        500,
        1,
        1
    ]

    # Restricciones
    restricciones = []

    # 1) A11 + A12 = G1  -> -G1 + A11 + A12 = 0
    restricciones.append({
        "coef": [-1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "signo": "=",
        "b": 0,
        "nombre": "Balance gas 1"
    })

    # 2) A21 + A22 = G2 -> -G2 + A21 + A22 = 0
    restricciones.append({
        "coef": [0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        "signo": "=",
        "b": 0,
        "nombre": "Balance gas 2"
    })

    # 3) Cada galón de gas 1 debe tener al menos 50% de aceite 1
    # A11 >= 0.5G1 -> 0.5G1 - A11 <= 0
    restricciones.append({
        "coef": [0.5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 0,
        "nombre": "Gas 1 mínimo 50% aceite 1"
    })

    # 4) Cada galón de gas 2 debe tener al menos 60% de aceite 1
    # A21 >= 0.6G2 -> 0.6G2 - A21 <= 0
    restricciones.append({
        "coef": [0, 0.6, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 0,
        "nombre": "Gas 2 mínimo 60% aceite 1"
    })

    # 5) Aceite 1 usado <= 500 disponibles + compras
    # A11 + A21 <= 500 + P1 + P2 + P3
    # A11 + A21 - P1 - P2 - P3 <= 500
    restricciones.append({
        "coef": [0, 0, 1, 0, 1, 0, -1, -1, -1, 0, 0],
        "signo": "<=",
        "b": 500,
        "nombre": "Disponibilidad aceite 1"
    })

    # 6) Aceite 2 usado <= 1000
    restricciones.append({
        "coef": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 1000,
        "nombre": "Disponibilidad aceite 2"
    })

    # 7) P1 <= 500
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "signo": "<=",
        "b": 500,
        "nombre": "Compra bloque 1 máximo 500"
    })

    # 8) P2 <= 500
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "signo": "<=",
        "b": 500,
        "nombre": "Compra bloque 2 máximo 500"
    })

    # 9) P3 <= 500
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "signo": "<=",
        "b": 500,
        "nombre": "Compra bloque 3 máximo 500"
    })

    # 10) Para usar bloque 2 debe activarse Y2
    # P2 <= 500Y2 -> P2 - 500Y2 <= 0
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, 0, 1, 0, -500, 0],
        "signo": "<=",
        "b": 0,
        "nombre": "Activación bloque 2"
    })

    # 11) Si se activa bloque 2, el bloque 1 debe llenarse
    # P1 >= 500Y2 -> -P1 + 500Y2 <= 0
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, -1, 0, 0, 500, 0],
        "signo": "<=",
        "b": 0,
        "nombre": "Llenar bloque 1 antes del bloque 2"
    })

    # 12) Para usar bloque 3, el bloque 2 debe llenarse
    # P3 <= 500Y3 y P2 >= 500Y3.
    # Aquí usamos P3 - 500Y3 <= 0.
    restricciones.append({
        "coef": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -500],
        "signo": "<=",
        "b": 0,
        "nombre": "Activación bloque 3"
    })

    # Guardar en session_state
    for i in range(11):
        st.session_state[f"nombre_{i}"] = nombres[i]
        st.session_state[f"tipo_var_{i}"] = tipos[i]
        st.session_state[f"c_{i}"] = float(c[i])
        st.session_state[f"lb_{i}"] = float(lb[i])
        st.session_state[f"ub_{i}"] = "" if ub[i] is None else float(ub[i])

    for r, rest in enumerate(restricciones):
        st.session_state[f"rest_nombre_{r}"] = rest["nombre"]
        st.session_state[f"signo_{r}"] = rest["signo"]
        st.session_state[f"b_{r}"] = float(rest["b"])
        for j in range(11):
            st.session_state[f"a_{r}_{j}"] = float(rest["coef"][j])


def preparar_matrices(n_vars, n_rest):
    nombres_vars = []
    tipos_vars = []
    c = []
    bounds = []

    for j in range(n_vars):
        nombre = st.session_state.get(f"nombre_{j}", f"x{j+1}")
        tipo = st.session_state.get(f"tipo_var_{j}", "Entera")
        coef = float(st.session_state.get(f"c_{j}", 0.0))

        lb_raw = st.session_state.get(f"lb_{j}", 0.0)
        ub_raw = st.session_state.get(f"ub_{j}", "")

        lb = 0.0 if lb_raw == "" else float(lb_raw)
        ub = None if ub_raw == "" else float(ub_raw)

        if tipo == "Binaria":
            lb = 0
            ub = 1

        nombres_vars.append(nombre)
        tipos_vars.append(tipo)
        c.append(coef)
        bounds.append((lb, ub))

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for i in range(n_rest):
        fila = []
        for j in range(n_vars):
            fila.append(float(st.session_state.get(f"a_{i}_{j}", 0.0)))

        signo = st.session_state.get(f"signo_{i}", "<=")
        b = float(st.session_state.get(f"b_{i}", 0.0))

        # Ignorar restricciones totalmente vacías
        if all(abs(x) < 1e-12 for x in fila) and abs(b) < 1e-12:
            continue

        if signo == "<=":
            A_ub.append(fila)
            b_ub.append(b)
        elif signo == ">=":
            A_ub.append([-x for x in fila])
            b_ub.append(-b)
        elif signo == "=":
            A_eq.append(fila)
            b_eq.append(b)

    return nombres_vars, tipos_vars, c, bounds, A_ub, b_ub, A_eq, b_eq


def resolver_lp(c_scipy, A_ub, b_ub, A_eq, b_eq, bounds):
    A_ub_param = A_ub if len(A_ub) > 0 else None
    b_ub_param = b_ub if len(b_ub) > 0 else None
    A_eq_param = A_eq if len(A_eq) > 0 else None
    b_eq_param = b_eq if len(b_eq) > 0 else None

    return linprog(
        c_scipy,
        A_ub=A_ub_param,
        b_ub=b_ub_param,
        A_eq=A_eq_param,
        b_eq=b_eq_param,
        bounds=bounds,
        method="highs"
    )


def dibujar_arbol(tree):
    def layout_arbol(graph, root=0, width=1.0, vert_gap=0.22, vert_loc=0, xcenter=0.5, pos=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        hijos = list(graph.successors(root))
        if len(hijos) > 0:
            dx = width / max(len(hijos), 2)
            next_x = xcenter - width / 2 - dx / 2
            for hijo in hijos:
                next_x += dx
                pos = layout_arbol(
                    graph,
                    root=hijo,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=next_x,
                    pos=pos
                )
        return pos

    if 0 not in tree.nodes:
        st.info("No hay árbol para mostrar.")
        return

    pos = layout_arbol(tree, 0)

    fig, ax = plt.subplots(figsize=(16, 10))

    node_colors = [tree.nodes[n].get("color", "white") for n in tree.nodes()]
    labels = nx.get_node_attributes(tree, "label")
    edge_labels = nx.get_edge_attributes(tree, "label")

    nx.draw_networkx_nodes(
        tree,
        pos,
        node_color=node_colors,
        node_size=3700,
        edgecolors="black",
        ax=ax
    )

    nx.draw_networkx_labels(
        tree,
        pos,
        labels=labels,
        font_size=8,
        ax=ax
    )

    nx.draw_networkx_edges(
        tree,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        ax=ax
    )

    nx.draw_networkx_edge_labels(
        tree,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        ax=ax
    )

    ax.axis("off")
    ax.margins(0.15)
    st.pyplot(fig)


# ============================================================
# FORMULACIÓN DEL EJERCICIO
# ============================================================
with st.expander("Ver formulación del ejercicio Euing Gas", expanded=False):
    st.markdown("""
### Variables

- **G1** = galones de gasolina 1 producidos  
- **G2** = galones de gasolina 2 producidos  
- **A11** = aceite 1 usado en gasolina 1  
- **A12** = aceite 2 usado en gasolina 1  
- **A21** = aceite 1 usado en gasolina 2  
- **A22** = aceite 2 usado en gasolina 2  
- **P1** = galones comprados de aceite 1 en el primer bloque, máximo 500 galones a 25 centavos  
- **P2** = galones comprados de aceite 1 en el segundo bloque, máximo 500 galones a 20 centavos  
- **P3** = galones comprados de aceite 1 en el tercer bloque, máximo 500 galones a 15 centavos  
- **Y2** = variable binaria para activar el segundo bloque de compra  
- **Y3** = variable binaria para activar el tercer bloque de compra  

### Función objetivo

Maximizar:

```text
Z = 12G1 + 14G2 - 25P1 - 20P2 - 15P3

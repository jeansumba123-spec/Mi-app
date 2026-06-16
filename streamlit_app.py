import streamlit as st
import math
import copy
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION GENERAL
# ============================================================
st.set_page_config(page_title="Euing Gas - Branch & Bound", layout="wide")

st.title("Optimizador de Programación Lineal Entera")
st.subheader("Branch & Bound con n variables y n restricciones")

st.markdown(
    "Esta aplicación resuelve problemas de programación lineal entera usando "
    "**Branch & Bound**. Incluye un botón para cargar automáticamente el ejercicio "
    "de **Euing Gas**."
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def es_entero(valor, tol=1e-6):
    return abs(valor - round(valor)) <= tol


def mejor_que(z_nuevo, z_actual, es_max):
    if es_max:
        return z_nuevo > z_actual + 1e-6
    return z_nuevo < z_actual - 1e-6


def peor_o_igual_que(z_nodo, z_mejor, es_max):
    if es_max:
        return z_nodo <= z_mejor + 1e-6
    return z_nodo >= z_mejor - 1e-6


def convertir_float(valor, defecto=0.0):
    try:
        if valor is None or valor == "":
            return defecto
        return float(valor)
    except Exception:
        return defecto


def convertir_cota_superior(valor):
    try:
        if valor is None or valor == "":
            return None
        return float(valor)
    except Exception:
        return None


def inicializar_euing_gas():
    st.session_state.tipo_opt = "Maximizar"
    st.session_state.n_vars = 11
    st.session_state.n_rest = 13

    nombres = ["G1", "G2", "A11", "A12", "A21", "A22", "P1", "P2", "P3", "Y2", "Y3"]

    tipos = [
        "Continua", "Continua", "Continua", "Continua", "Continua", "Continua",
        "Continua", "Continua", "Continua", "Binaria", "Binaria"
    ]

    # Objetivo en centavos:
    # Max Z = 12G1 + 14G2 - 25P1 - 20P2 - 15P3
    c = [12, 14, 0, 0, 0, 0, -25, -20, -15, 0, 0]

    lb = [0] * 11
    ub = [None, None, None, None, None, None, 500, 500, 500, 1, 1]

    restricciones = []

    restricciones.append({
        "nombre": "Balance gasolina 1",
        "coef": [-1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "signo": "=",
        "b": 0
    })

    restricciones.append({
        "nombre": "Balance gasolina 2",
        "coef": [0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        "signo": "=",
        "b": 0
    })

    restricciones.append({
        "nombre": "Gas 1 con mínimo 50% de aceite 1",
        "coef": [0.5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 0
    })

    restricciones.append({
        "nombre": "Gas 2 con mínimo 60% de aceite 1",
        "coef": [0, 0.6, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 0
    })

    restricciones.append({
        "nombre": "Disponibilidad de aceite 1",
        "coef": [0, 0, 1, 0, 1, 0, -1, -1, -1, 0, 0],
        "signo": "<=",
        "b": 500
    })

    restricciones.append({
        "nombre": "Disponibilidad de aceite 2",
        "coef": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "signo": "<=",
        "b": 1000
    })

    restricciones.append({
        "nombre": "P1 máximo 500",
        "coef": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "signo": "<=",
        "b": 500
    })

    restricciones.append({
        "nombre": "P2 máximo 500",
        "coef": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "signo": "<=",
        "b": 500
    })

    restricciones.append({
        "nombre": "P3 máximo 500",
        "coef": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "signo": "<=",
        "b": 500
    })

    # P2 <= 500Y2
    restricciones.append({
        "nombre": "Activar bloque 2",
        "coef": [0, 0, 0, 0, 0, 0, 0, 1, 0, -500, 0],
        "signo": "<=",
        "b": 0
    })

    # P1 >= 500Y2  -> -P1 + 500Y2 <= 0
    restricciones.append({
        "nombre": "Llenar bloque 1 antes del bloque 2",
        "coef": [0, 0, 0, 0, 0, 0, -1, 0, 0, 500, 0],
        "signo": "<=",
        "b": 0
    })

    # P3 <= 500Y3
    restricciones.append({
        "nombre": "Activar bloque 3",
        "coef": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -500],
        "signo": "<=",
        "b": 0
    })

    # P2 >= 500Y3 -> -P2 + 500Y3 <= 0
    restricciones.append({
        "nombre": "Llenar bloque 2 antes del bloque 3",
        "coef": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 500],
        "signo": "<=",
        "b": 0
    })

    for j in range(11):
        st.session_state[f"nombre_{j}"] = nombres[j]
        st.session_state[f"tipo_var_{j}"] = tipos[j]
        st.session_state[f"c_{j}"] = float(c[j])
        st.session_state[f"lb_{j}"] = str(lb[j])
        st.session_state[f"ub_{j}"] = "" if ub[j] is None else str(ub[j])

    for i, rest in enumerate(restricciones):
        st.session_state[f"rest_nombre_{i}"] = rest["nombre"]
        st.session_state[f"signo_{i}"] = rest["signo"]
        st.session_state[f"b_{i}"] = float(rest["b"])
        for j in range(11):
            st.session_state[f"a_{i}_{j}"] = float(rest["coef"][j])


def preparar_matrices(n_vars, n_rest):
    nombres_vars = []
    tipos_vars = []
    c = []
    bounds = []

    for j in range(n_vars):
        nombre = st.session_state.get(f"nombre_{j}", f"x{j + 1}")
        tipo = st.session_state.get(f"tipo_var_{j}", "Continua")
        coef = convertir_float(st.session_state.get(f"c_{j}", 0.0))

        lb = convertir_float(st.session_state.get(f"lb_{j}", 0.0), 0.0)
        ub = convertir_cota_superior(st.session_state.get(f"ub_{j}", ""))

        if tipo == "Binaria":
            lb = 0.0
            ub = 1.0

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
            fila.append(convertir_float(st.session_state.get(f"a_{i}_{j}", 0.0)))

        signo = st.session_state.get(f"signo_{i}", "<=")
        b = convertir_float(st.session_state.get(f"b_{i}", 0.0))

        if all(abs(x) < 1e-12 for x in fila) and abs(b) < 1e-12:
            continue

        if signo == "<=":
            A_ub.append(fila)
            b_ub.append(b)
        elif signo == ">=":
            A_ub.append([-x for x in fila])
            b_ub.append(-b)
        else:
            A_eq.append(fila)
            b_eq.append(b)

    return nombres_vars, tipos_vars, c, bounds, A_ub, b_ub, A_eq, b_eq


def resolver_lp(c_scipy, A_ub, b_ub, A_eq, b_eq, bounds):
    return linprog(
        c_scipy,
        A_ub=A_ub if len(A_ub) > 0 else None,
        b_ub=b_ub if len(b_ub) > 0 else None,
        A_eq=A_eq if len(A_eq) > 0 else None,
        b_eq=b_eq if len(b_eq) > 0 else None,
        bounds=bounds,
        method="highs"
    )


def dibujar_arbol(tree):
    if len(tree.nodes()) == 0 or 0 not in tree.nodes:
        st.info("No hay árbol para mostrar.")
        return

    def jerarquia(graph, nodo, x=0.5, y=0, ancho=1.0, dy=0.25, pos=None):
        if pos is None:
            pos = {}
        pos[nodo] = (x, y)
        hijos = list(graph.successors(nodo))

        if len(hijos) == 1:
            jerarquia(graph, hijos[0], x, y - dy, ancho / 2, dy, pos)
        elif len(hijos) >= 2:
            jerarquia(graph, hijos[0], x - ancho / 4, y - dy, ancho / 2, dy, pos)
            jerarquia(graph, hijos[1], x + ancho / 4, y - dy, ancho / 2, dy, pos)

        return pos

    pos = jerarquia(tree, 0)

    fig, ax = plt.subplots(figsize=(16, 9))

    colores = [tree.nodes[n].get("color", "white") for n in tree.nodes()]
    etiquetas = nx.get_node_attributes(tree, "label")
    etiquetas_aristas = nx.get_edge_attributes(tree, "label")

    nx.draw_networkx_nodes(
        tree,
        pos,
        node_color=colores,
        node_size=3600,
        edgecolors="black",
        ax=ax
    )
    nx.draw_networkx_labels(
        tree,
        pos,
        labels=etiquetas,
        font_size=8,
        ax=ax
    )
    nx.draw_networkx_edges(
        tree,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        ax=ax
    )
    nx.draw_networkx_edge_labels(
        tree,
        pos,
        edge_labels=etiquetas_aristas,
        font_size=8,
        rotate=False,
        ax=ax
    )

    ax.axis("off")
    ax.margins(0.15)
    st.pyplot(fig)


# ============================================================
# FORMULACION DEL EJERCICIO
# ============================================================
with st.expander("Ver formulación del ejercicio Euing Gas", expanded=False):
    st.markdown("### Variables")
    st.markdown(
        "- G1: galones de gasolina 1 producidos\n"
        "- G2: galones de gasolina 2 producidos\n"
        "- A11: aceite 1 usado en gasolina 1\n"
        "- A12: aceite 2 usado en gasolina 1\n"
        "- A21: aceite 1 usado en gasolina 2\n"
        "- A22: aceite 2 usado en gasolina 2\n"
        "- P1: compra de aceite 1 en el primer bloque, a 25 centavos\n"
        "- P2: compra de aceite 1 en el segundo bloque, a 20 centavos\n"
        "- P3: compra de aceite 1 en el tercer bloque, a 15 centavos\n"
        "- Y2: variable binaria para activar bloque 2\n"
        "- Y3: variable binaria para activar bloque 3"
    )

    st.markdown("### Función objetivo")
    st.latex(r"Max\ Z = 12G1 + 14G2 - 25P1 - 20P2 - 15P3")

    st.markdown("### Restricciones principales")
    st.markdown(
        "- Balance de mezcla: el aceite usado debe igualar la gasolina producida.\n"
        "- Gasolina 1: al menos 50% de aceite 1.\n"
        "- Gasolina 2: al menos 60% de aceite 1.\n"
        "- Aceite 1 disponible: 500 galones más compras adicionales.\n"
        "- Aceite 2 disponible: 1000 galones.\n"
        "- Las compras de aceite 1 se dividen en tres bloques de 500 galones."
    )


# ============================================================
# ENTRADA DE DATOS
# ============================================================
st.header("1. Datos del problema")

col_boton, col_info = st.columns([1, 3])

with col_boton:
    if st.button("Cargar ejercicio Euing Gas"):
        inicializar_euing_gas()
        st.success("Ejercicio cargado. Ahora presiona Ejecutar Branch & Bound.")

with col_info:
    st.info("Puedes usar el ejercicio cargado o modificar variables y restricciones manualmente.")

tipo_opt = st.selectbox(
    "Tipo de optimización",
    ["Maximizar", "Minimizar"],
    key="tipo_opt"
)

col1, col2, col3 = st.columns(3)

with col1:
    n_vars = st.number_input(
        "Número de variables",
        min_value=1,
        max_value=30,
        value=int(st.session_state.get("n_vars", 3)),
        step=1,
        key="n_vars"
    )

with col2:
    n_rest = st.number_input(
        "Número de restricciones",
        min_value=1,
        max_value=100,
        value=int(st.session_state.get("n_rest", 3)),
        step=1,
        key="n_rest"
    )

with col3:
    max_nodos = st.number_input(
        "Máximo de nodos",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )


# ============================================================
# VARIABLES
# ============================================================
st.header("2. Variables y función objetivo")

st.caption("La cota superior puede quedar vacía si no existe límite superior.")

for j in range(int(n_vars)):
    cols = st.columns([1.3, 1.2, 1.2, 1.2, 1.2])

    with cols[0]:
        st.text_input(
            f"Nombre variable {j + 1}",
            value=st.session_state.get(f"nombre_{j}", f"x{j + 1}"),
            key=f"nombre_{j}"
        )

    with cols[1]:
        valor_tipo = st.session_state.get(f"tipo_var_{j}", "Continua")
        if valor_tipo not in ["Continua", "Entera", "Binaria"]:
            valor_tipo = "Continua"
        st.selectbox(
            f"Tipo {j + 1}",
            ["Continua", "Entera", "Binaria"],
            index=["Continua", "Entera", "Binaria"].index(valor_tipo),
            key=f"tipo_var_{j}"
        )

    with cols[2]:
        st.number_input(
            f"Coeficiente Z {j + 1}",
            value=float(st.session_state.get(f"c_{j}", 0.0)),
            step=1.0,
            key=f"c_{j}"
        )

    with cols[3]:
        st.text_input(
            f"Cota inferior {j + 1}",
            value=str(st.session_state.get(f"lb_{j}", 0.0)),
            key=f"lb_{j}"
        )

    with cols[4]:
        st.text_input(
            f"Cota superior {j + 1}",
            value=str(st.session_state.get(f"ub_{j}", "")),
            key=f"ub_{j}"
        )


# ============================================================
# RESTRICCIONES
# ============================================================
st.header("3. Restricciones")

for i in range(int(n_rest)):
    with st.expander(f"Restricción {i + 1}", expanded=True):
        st.text_input(
            f"Nombre R{i + 1}",
            value=st.session_state.get(f"rest_nombre_{i}", f"R{i + 1}"),
            key=f"rest_nombre_{i}"
        )

        inicio = 0
        while inicio < int(n_vars):
            fin = min(inicio + 6, int(n_vars))
            columnas = st.columns(fin - inicio)

            for idx, j in enumerate(range(inicio, fin)):
                with columnas[idx]:
                    nombre_var = st.session_state.get(f"nombre_{j}", f"x{j + 1}")
                    st.number_input(
                        f"{nombre_var} en R{i + 1}",
                        value=float(st.session_state.get(f"a_{i}_{j}", 0.0)),
                        step=1.0,
                        key=f"a_{i}_{j}"
                    )

            inicio = fin

        col_s, col_b = st.columns([1, 2])

        with col_s:
            signo_actual = st.session_state.get(f"signo_{i}", "<=")
            if signo_actual not in ["<=", ">=", "="]:
                signo_actual = "<="
            st.selectbox(
                f"Signo R{i + 1}",
                ["<=", ">=", "="],
                index=["<=", ">=", "="].index(signo_actual),
                key=f"signo_{i}"
            )

        with col_b:
            st.number_input(
                f"Lado derecho R{i + 1}",
                value=float(st.session_state.get(f"b_{i}", 0.0)),
                step=1.0,
                key=f"b_{i}"
            )


# ============================================================
# EJECUCION BRANCH AND BOUND
# ============================================================
st.header("4. Ejecutar")

if st.button("Ejecutar Branch & Bound", type="primary"):
    es_max = tipo_opt == "Maximizar"

    nombres_vars, tipos_vars, c_original, bounds, A_ub, b_ub, A_eq, b_eq = preparar_matrices(
        int(n_vars),
        int(n_rest)
    )

    c_scipy = [-coef for coef in c_original] if es_max else c_original[:]

    mejor_z = -float("inf") if es_max else float("inf")
    mejor_x = None

    consola = []
    tree = nx.DiGraph()

    nodos_pila = [{
        "id": 0,
        "A_ub": copy.deepcopy(A_ub),
        "b_ub": copy.deepcopy(b_ub),
        "A_eq": copy.deepcopy(A_eq),
        "b_eq": copy.deepcopy(b_eq),
        "parent": None,
        "edge": "Raíz"
    }]

    contador_nodos = 0
    nodos_explorados = 0

    while nodos_pila and nodos_explorados < int(max_nodos):
        nodo = nodos_pila.pop()
        nodos_explorados += 1

        tree.add_node(nodo["id"], label=f"N{nodo['id']}", color="#BFD7EA")

        if nodo["parent"] is not None:
            tree.add_edge(nodo["parent"], nodo["id"], label=nodo["edge"])

        consola.append(f"Explorando Nodo {nodo['id']}")

        res = resolver_lp(
            c_scipy,
            nodo["A_ub"],
            nodo["b_ub"],
            nodo["A_eq"],
            nodo["b_eq"],
            bounds
        )

        if not res.success:
            consola.append("  Podado: problema infactible.\n")
            tree.nodes[nodo["id"]]["color"] = "#FF9999"
            tree.nodes[nodo["id"]]["label"] = f"N{nodo['id']}\nInfactible"
            continue

        x = res.x
        z = -res.fun if es_max else res.fun

        consola.append(f"  Z relajado = {z:.4f}")
        consola.append(
            "  Solución relajada: " +
            ", ".join([f"{nombres_vars[j]}={x[j]:.4f}" for j in range(int(n_vars))])
        )

        if mejor_x is not None and peor_o_igual_que(z, mejor_z, es_max):
            consola.append(f"  Podado por cota. Mejor Z actual = {mejor_z:.4f}\n")
            tree.nodes[nodo["id"]]["color"] = "#FFD699"
            tree.nodes[nodo["id"]]["label"] = f"N{nodo['id']}\nZ={z:.2f}\nPodado"
            continue

        idx_frac = None
        val_frac = None

        for j in range(int(n_vars)):
            if tipos_vars[j] in ["Entera", "Binaria"] and not es_entero(x[j]):
                idx_frac = j
                val_frac = x[j]
                break

        if idx_frac is None:
            if mejor_x is None or mejor_que(z, mejor_z, es_max):
                mejor_z = z
                mejor_x = x.copy()

            consola.append("  Solución entera factible encontrada.")
            consola.append(f"  Mejor Z actualizado = {mejor_z:.4f}\n")

            tree.nodes[nodo["id"]]["color"] = "#99FF99"
            tree.nodes[nodo["id"]]["label"] = f"N{nodo['id']}\nZ={z:.2f}\nEntero"
            continue

        nombre_var = nombres_vars[idx_frac]
        piso = math.floor(val_frac)
        techo = math.ceil(val_frac)

        consola.append(f"  Variable fraccionaria: {nombre_var} = {val_frac:.4f}")
        consola.append(f"  Ramificación: {nombre_var} <= {piso} y {nombre_var} >= {techo}\n")

        tree.nodes[nodo["id"]]["color"] = "#99CCFF"
        tree.nodes[nodo["id"]]["label"] = f"N{nodo['id']}\nZ={z:.2f}\nBranch"

        # Rama izquierda: x <= floor(valor)
        A_izq = copy.deepcopy(nodo["A_ub"])
        b_izq = copy.deepcopy(nodo["b_ub"])
        fila_izq = [0.0] * int(n_vars)
        fila_izq[idx_frac] = 1.0
        A_izq.append(fila_izq)
        b_izq.append(float(piso))

        contador_nodos += 1
        nodos_pila.append({
            "id": contador_nodos,
            "A_ub": A_izq,
            "b_ub": b_izq,
            "A_eq": copy.deepcopy(nodo["A_eq"]),
            "b_eq": copy.deepcopy(nodo["b_eq"]),
            "parent": nodo["id"],
            "edge": f"{nombre_var} <= {piso}"
        })

        # Rama derecha: x >= ceil(valor), equivalente a -x <= -ceil(valor)
        A_der = copy.deepcopy(nodo["A_ub"])
        b_der = copy.deepcopy(nodo["b_ub"])
        fila_der = [0.0] * int(n_vars)
        fila_der[idx_frac] = -1.0
        A_der.append(fila_der)
        b_der.append(float(-techo))

        contador_nodos += 1
        nodos_pila.append({
            "id": contador_nodos,
            "A_ub": A_der,
            "b_ub": b_der,
            "A_eq": copy.deepcopy(nodo["A_eq"]),
            "b_eq": copy.deepcopy(nodo["b_eq"]),
            "parent": nodo["id"],
            "edge": f"{nombre_var} >= {techo}"
        })

    st.header("5. Historial de Branch & Bound")
    st.code("\n".join(consola), language="text")

    if nodos_explorados >= int(max_nodos):
        st.warning("Se alcanzó el límite máximo de nodos. Puedes aumentar el límite.")

    st.header("6. Solución final")

    if mejor_x is None:
        st.error("No se encontró una solución entera factible.")
    else:
        st.success(f"Mejor valor de Z encontrado: {mejor_z:.4f} centavos")

        tabla = []
        for j in range(int(n_vars)):
            tabla.append({
                "Variable": nombres_vars[j],
                "Tipo": tipos_vars[j],
                "Valor": round(float(mejor_x[j]), 6)
            })

        st.dataframe(tabla, use_container_width=True)

        if int(n_vars) == 11:
            st.markdown("### Interpretación del ejercicio Euing Gas")
            st.write(f"Gasolina 1 producida: **{mejor_x[0]:.4f} galones**")
            st.write(f"Gasolina 2 producida: **{mejor_x[1]:.4f} galones**")
            st.write(f"Aceite 1 usado en gasolina 1: **{mejor_x[2]:.4f} galones**")
            st.write(f"Aceite 2 usado en gasolina 1: **{mejor_x[3]:.4f} galones**")
            st.write(f"Aceite 1 usado en gasolina 2: **{mejor_x[4]:.4f} galones**")
            st.write(f"Aceite 2 usado en gasolina 2: **{mejor_x[5]:.4f} galones**")
            st.write(f"Compra bloque 1: **{mejor_x[6]:.4f} galones**")
            st.write(f"Compra bloque 2: **{mejor_x[7]:.4f} galones**")
            st.write(f"Compra bloque 3: **{mejor_x[8]:.4f} galones**")

    st.header("7. Árbol de ramificación")
    st.markdown(
        "- Azul: nodo ramificado\n"
        "- Verde: solución entera factible\n"
        "- Rojo: nodo infactible\n"
        "- Naranja: podado por cota"
    )

    dibujar_arbol(tree)

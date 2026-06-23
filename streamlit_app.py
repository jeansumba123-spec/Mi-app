import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math


# ============================================================
# CONFIGURACION GENERAL
# ============================================================
st.set_page_config(page_title="Ruta más corta - Dijkstra", layout="wide")

st.title("Optimizador de Ruta Más Corta")
st.subheader("Algoritmo de Dijkstra con tablas iterativas")

st.markdown(
    "Este programa permite resolver el problema de la ruta más corta desde un nodo origen "
    "hasta un nodo destino. Incluye el ejercicio de la figura desde **A hasta J**, pero también "
    "puedes ingresar otro grafo personalizado."
)


# ============================================================
# DATOS DEL EJERCICIO DE LA IMAGEN
# ============================================================
def cargar_ejercicio_imagen():
    st.session_state.nodos_texto = "A,B,C,D,E,F,G,H,I,J"
    st.session_state.origen = "A"
    st.session_state.destino = "J"
    st.session_state.dirigido = True

    st.session_state.aristas_texto = """A,B,2
A,C,4
A,D,3
B,E,7
B,F,4
B,G,6
C,E,3
C,F,2
C,G,4
D,E,4
D,F,1
D,G,5
E,H,1
E,I,4
F,H,6
F,I,3
G,H,3
G,I,3
H,J,3
I,J,4"""


# ============================================================
# FUNCIONES
# ============================================================
def limpiar_nodo(x):
    return str(x).strip().upper()


def leer_nodos(nodos_texto):
    nodos = []
    partes = nodos_texto.replace("\n", ",").split(",")

    for p in partes:
        nodo = limpiar_nodo(p)
        if nodo != "" and nodo not in nodos:
            nodos.append(nodo)

    return nodos


def leer_aristas(aristas_texto):
    aristas = []
    errores = []

    lineas = aristas_texto.strip().splitlines()

    for i, linea in enumerate(lineas, start=1):
        linea = linea.strip()

        if linea == "":
            continue

        partes = linea.split(",")

        if len(partes) != 3:
            errores.append(f"Línea {i}: debe tener formato Origen,Destino,Costo")
            continue

        u = limpiar_nodo(partes[0])
        v = limpiar_nodo(partes[1])

        try:
            costo = float(partes[2])
        except Exception:
            errores.append(f"Línea {i}: el costo no es numérico")
            continue

        if costo < 0:
            errores.append(f"Línea {i}: Dijkstra no permite costos negativos")
            continue

        aristas.append((u, v, costo))

    return aristas, errores


def construir_grafo(nodos, aristas, dirigido=True):
    if dirigido:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(nodos)

    for u, v, costo in aristas:
        G.add_edge(u, v, weight=costo)

        if u not in G.nodes:
            G.add_node(u)
        if v not in G.nodes:
            G.add_node(v)

    return G


def dijkstra_con_tablas(nodos, aristas, origen, destino, dirigido=True):
    adj = {n: [] for n in nodos}

    for u, v, costo in aristas:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []

        adj[u].append((v, costo))

        if not dirigido:
            adj[v].append((u, costo))

    dist = {n: math.inf for n in adj.keys()}
    prev = {n: None for n in adj.keys()}
    definitivo = {n: False for n in adj.keys()}

    dist[origen] = 0

    cola = [(0, origen)]
    tablas = []
    historial = []
    iteracion = 0

    while cola:
        distancia_actual, nodo_actual = heapq.heappop(cola)

        if definitivo[nodo_actual]:
            continue

        definitivo[nodo_actual] = True
        iteracion += 1

        historial.append(
            f"Iteración {iteracion}: se selecciona el nodo {nodo_actual} "
            f"con distancia acumulada {distancia_actual:g}."
        )

        actualizaciones = []

        for vecino, costo in adj.get(nodo_actual, []):
            if definitivo[vecino]:
                continue

            nueva_distancia = dist[nodo_actual] + costo

            if nueva_distancia < dist[vecino]:
                dist[vecino] = nueva_distancia
                prev[vecino] = nodo_actual
                heapq.heappush(cola, (nueva_distancia, vecino))

                actualizaciones.append(
                    f"{vecino}: nueva distancia {nueva_distancia:g}, "
                    f"predecesor {nodo_actual}"
                )

        if len(actualizaciones) == 0:
            historial.append("  No hubo actualizaciones.")
        else:
            for act in actualizaciones:
                historial.append("  " + act)

        tabla = []

        for n in sorted(adj.keys()):
            if dist[n] == math.inf:
                distancia_mostrar = "∞"
            else:
                distancia_mostrar = dist[n]

            tabla.append({
                "Nodo": n,
                "Distancia tentativa": distancia_mostrar,
                "Predecesor": "-" if prev[n] is None else prev[n],
                "Definitivo": "Sí" if definitivo[n] else "No"
            })

        tablas.append({
            "iteracion": iteracion,
            "nodo_actual": nodo_actual,
            "tabla": pd.DataFrame(tabla)
        })

        if nodo_actual == destino:
            break

    if dist.get(destino, math.inf) == math.inf:
        return None, math.inf, tablas, historial, dist, prev

    ruta = []
    actual = destino

    while actual is not None:
        ruta.append(actual)
        actual = prev[actual]

    ruta.reverse()

    return ruta, dist[destino], tablas, historial, dist, prev


def dibujar_grafo(G, ruta=None):
    fig, ax = plt.subplots(figsize=(13, 7))

    try:
        pos = nx.multipartite_layout(
            G,
            subset_key=lambda n: {
                "A": 0,
                "B": 1,
                "C": 1,
                "D": 1,
                "E": 2,
                "F": 2,
                "G": 2,
                "H": 3,
                "I": 3,
                "J": 4
            }.get(n, 0)
        )
    except Exception:
        pos = nx.spring_layout(G, seed=7)

    colores_nodos = []

    if ruta is None:
        ruta = []

    for n in G.nodes:
        if n in ruta:
            colores_nodos.append("#99FF99")
        else:
            colores_nodos.append("#D9EAF7")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1300,
        node_color=colores_nodos,
        edgecolors="black",
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.5,
        ax=ax
    )

    edge_labels = nx.get_edge_attributes(G, "weight")

    edge_labels_limpios = {}
    for k, v in edge_labels.items():
        if float(v).is_integer():
            edge_labels_limpios[k] = str(int(v))
        else:
            edge_labels_limpios[k] = str(v)

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels_limpios,
        font_size=11,
        ax=ax
    )

    if ruta is not None and len(ruta) > 1:
        aristas_ruta = []

        for i in range(len(ruta) - 1):
            aristas_ruta.append((ruta[i], ruta[i + 1]))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=aristas_ruta,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            width=4,
            edge_color="#00AA00",
            ax=ax
        )

    ax.axis("off")
    st.pyplot(fig)


# ============================================================
# VALORES INICIALES
# ============================================================
if "nodos_texto" not in st.session_state:
    cargar_ejercicio_imagen()


# ============================================================
# ENTRADA DE DATOS
# ============================================================
st.header("1. Definir el grafo")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Cargar ejercicio de la imagen"):
        cargar_ejercicio_imagen()
        st.success("Ejercicio cargado.")

with col2:
    st.info(
        "Puedes usar el ejercicio de la imagen o modificar los nodos y caminos "
        "para resolver otro problema."
    )

st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    origen = st.text_input(
        "Nodo origen",
        value=st.session_state.get("origen", "A"),
        key="origen"
    )

with col_b:
    destino = st.text_input(
        "Nodo destino",
        value=st.session_state.get("destino", "J"),
        key="destino"
    )

with col_c:
    dirigido = st.checkbox(
        "Grafo dirigido",
        value=st.session_state.get("dirigido", True),
        key="dirigido"
    )

nodos_texto = st.text_area(
    "Nodos separados por coma",
    value=st.session_state.get("nodos_texto", "A,B,C,D,E,F,G,H,I,J"),
    height=80,
    key="nodos_texto"
)

st.markdown("### Caminos o aristas")

st.markdown(
    "Escribe cada camino en una línea con el formato:"
)

st.code("Origen,Destino,Costo", language="text")

aristas_texto = st.text_area(
    "Lista de caminos",
    value=st.session_state.get("aristas_texto", ""),
    height=300,
    key="aristas_texto"
)


# ============================================================
# PROCESAR DATOS
# ============================================================
nodos = leer_nodos(nodos_texto)
aristas, errores = leer_aristas(aristas_texto)

origen_limpio = limpiar_nodo(origen)
destino_limpio = limpiar_nodo(destino)

for u, v, c in aristas:
    if u not in nodos:
        nodos.append(u)
    if v not in nodos:
        nodos.append(v)


if errores:
    st.error("Hay errores en los datos ingresados:")
    for e in errores:
        st.write(e)


# ============================================================
# MOSTRAR DATOS CARGADOS
# ============================================================
st.header("2. Datos cargados")

col_datos1, col_datos2 = st.columns(2)

with col_datos1:
    st.markdown("### Nodos")
    st.write(", ".join(nodos))

with col_datos2:
    st.markdown("### Caminos")
    if len(aristas) > 0:
        df_aristas = pd.DataFrame(aristas, columns=["Origen", "Destino", "Costo"])
        st.dataframe(df_aristas, use_container_width=True)
    else:
        st.warning("No hay caminos cargados.")


# ============================================================
# EJECUCION
# ============================================================
st.header("3. Resolver ruta más corta")

if st.button("Calcular ruta más corta", type="primary"):
    if len(errores) > 0:
        st.error("Corrige los errores antes de ejecutar.")
    elif origen_limpio not in nodos:
        st.error(f"El nodo origen {origen_limpio} no existe.")
    elif destino_limpio not in nodos:
        st.error(f"El nodo destino {destino_limpio} no existe.")
    else:
        G = construir_grafo(nodos, aristas, dirigido=dirigido)

        ruta, costo, tablas, historial, dist, prev = dijkstra_con_tablas(
            nodos=nodos,
            aristas=aristas,
            origen=origen_limpio,
            destino=destino_limpio,
            dirigido=dirigido
        )

        st.header("4. Resultado final")

        if ruta is None:
            st.error(f"No existe una ruta desde {origen_limpio} hasta {destino_limpio}.")
            dibujar_grafo(G, ruta=[])
        else:
            st.success(f"Ruta más corta: {' → '.join(ruta)}")
            st.success(f"Costo mínimo total: {costo:g}")

            st.markdown("### Interpretación")
            st.write(
                f"Para ir desde **{origen_limpio}** hasta **{destino_limpio}**, "
                f"la ruta recomendada es **{' → '.join(ruta)}**, con un costo total de **{costo:g}**."
            )

            st.header("5. Grafo con la ruta resaltada")
            dibujar_grafo(G, ruta=ruta)

            st.header("6. Historial del algoritmo")
            st.code("\n".join(historial), language="text")

            st.header("7. Tablas iterativas")

            for item in tablas:
                st.markdown(
                    f"### Iteración {item['iteracion']} - Nodo seleccionado: {item['nodo_actual']}"
                )
                st.dataframe(item["tabla"], use_container_width=True)

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Ruta más corta - Dijkstra",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Optimizador de Ruta Más Corta")
st.markdown("### Algoritmo de Dijkstra con tablas iterativas y grafo visual")

st.markdown(
    """
Esta aplicación permite resolver problemas de **ruta más corta** desde un nodo origen hasta un nodo destino.

El programa muestra:

- Los caminos ingresados.
- El grafo completo.
- La ruta más corta resaltada.
- El costo mínimo total.
- El historial del algoritmo.
- Cada tabla iterativa de Dijkstra.
"""
)


# ============================================================
# DATOS DEL EJERCICIO DE LA IMAGEN
# ============================================================
def datos_ejercicio_imagen():
    return pd.DataFrame(
        [
            ["A", "B", 2],
            ["A", "C", 4],
            ["A", "D", 3],
            ["B", "E", 7],
            ["B", "F", 4],
            ["B", "G", 6],
            ["C", "E", 3],
            ["C", "F", 2],
            ["C", "G", 4],
            ["D", "E", 4],
            ["D", "F", 1],
            ["D", "G", 5],
            ["E", "H", 1],
            ["E", "I", 4],
            ["F", "H", 6],
            ["F", "I", 3],
            ["G", "H", 3],
            ["G", "I", 3],
            ["H", "J", 3],
            ["I", "J", 4],
        ],
        columns=["Origen", "Destino", "Costo"]
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def limpiar_nodo(valor):
    return str(valor).strip().upper()


def formatear_numero(x):
    if x == math.inf:
        return "∞"
    try:
        if float(x).is_integer():
            return int(x)
        return round(float(x), 4)
    except Exception:
        return x


def procesar_aristas(df):
    aristas = []
    errores = []

    if df is None or len(df) == 0:
        return aristas, ["No se ingresaron caminos."]

    for idx, fila in df.iterrows():
        origen = limpiar_nodo(fila.get("Origen", ""))
        destino = limpiar_nodo(fila.get("Destino", ""))
        costo_raw = fila.get("Costo", "")

        if origen == "" and destino == "" and (pd.isna(costo_raw) or costo_raw == ""):
            continue

        if origen == "":
            errores.append(f"Fila {idx + 1}: falta el nodo origen.")
            continue

        if destino == "":
            errores.append(f"Fila {idx + 1}: falta el nodo destino.")
            continue

        try:
            costo = float(costo_raw)
        except Exception:
            errores.append(f"Fila {idx + 1}: el costo debe ser numérico.")
            continue

        if costo < 0:
            errores.append(
                f"Fila {idx + 1}: el algoritmo de Dijkstra no permite costos negativos."
            )
            continue

        aristas.append((origen, destino, costo))

    if len(aristas) == 0:
        errores.append("No hay caminos válidos para resolver.")

    return aristas, errores


def obtener_nodos(aristas, origen, destino):
    nodos = []

    for u, v, _ in aristas:
        if u not in nodos:
            nodos.append(u)
        if v not in nodos:
            nodos.append(v)

    if origen and origen not in nodos:
        nodos.append(origen)

    if destino and destino not in nodos:
        nodos.append(destino)

    return sorted(nodos)


def construir_grafo(aristas, dirigido=True):
    G = nx.DiGraph() if dirigido else nx.Graph()

    for u, v, costo in aristas:
        G.add_edge(u, v, weight=costo)

    return G


def dijkstra_con_tablas(aristas, origen, destino, dirigido=True):
    nodos = obtener_nodos(aristas, origen, destino)

    adyacencia = {nodo: [] for nodo in nodos}

    for u, v, costo in aristas:
        adyacencia[u].append((v, costo))

        if not dirigido:
            adyacencia[v].append((u, costo))

    distancia = {nodo: math.inf for nodo in nodos}
    predecesor = {nodo: None for nodo in nodos}
    definitivo = {nodo: False for nodo in nodos}

    distancia[origen] = 0

    cola = [(0, origen)]
    iteracion = 0
    historial = []
    tablas = []

    while cola:
        distancia_actual, nodo_actual = heapq.heappop(cola)

        if definitivo[nodo_actual]:
            continue

        definitivo[nodo_actual] = True
        iteracion += 1

        historial.append(
            f"Iteración {iteracion}: se selecciona el nodo {nodo_actual} "
            f"con distancia acumulada {formatear_numero(distancia_actual)}."
        )

        actualizaciones = []

        for vecino, costo in adyacencia.get(nodo_actual, []):
            if definitivo[vecino]:
                continue

            nueva_distancia = distancia[nodo_actual] + costo

            if nueva_distancia < distancia[vecino]:
                distancia[vecino] = nueva_distancia
                predecesor[vecino] = nodo_actual
                heapq.heappush(cola, (nueva_distancia, vecino))

                actualizaciones.append(
                    f"{vecino}: distancia = {formatear_numero(nueva_distancia)}, "
                    f"predecesor = {nodo_actual}"
                )

        if actualizaciones:
            for act in actualizaciones:
                historial.append("  " + act)
        else:
            historial.append("  No hubo actualizaciones.")

        tabla_iteracion = []

        for nodo in nodos:
            tabla_iteracion.append(
                {
                    "Nodo": nodo,
                    "Distancia tentativa": formatear_numero(distancia[nodo]),
                    "Predecesor": "-" if predecesor[nodo] is None else predecesor[nodo],
                    "Definitivo": "Sí" if definitivo[nodo] else "No",
                }
            )

        tablas.append(
            {
                "iteracion": iteracion,
                "nodo_actual": nodo_actual,
                "tabla": pd.DataFrame(tabla_iteracion),
            }
        )

        if nodo_actual == destino:
            break

    if distancia[destino] == math.inf:
        return None, math.inf, tablas, historial, distancia, predecesor

    ruta = []
    actual = destino

    while actual is not None:
        ruta.append(actual)
        actual = predecesor[actual]

    ruta.reverse()

    return ruta, distancia[destino], tablas, historial, distancia, predecesor


# ============================================================
# POSICIONES BONITAS DEL GRAFO
# ============================================================
def posiciones_grafo(G, origen, destino):
    nodos = list(G.nodes())

    # Posiciones exactas para el ejercicio de la imagen
    nodos_imagen = set(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

    if set(nodos).issubset(nodos_imagen) and "A" in nodos and "J" in nodos:
        pos_base = {
            "A": (0, 0),
            "B": (2, 1.6),
            "C": (2, 0),
            "D": (2, -1.6),
            "E": (4.2, 1.6),
            "F": (4.2, 0),
            "G": (4.2, -1.6),
            "H": (6.4, 0.9),
            "I": (6.4, -0.9),
            "J": (8.3, 0),
        }
        return {n: pos_base[n] for n in nodos if n in pos_base}

    # Layout automático para otros grafos
    try:
        niveles = {}
        longitudes = nx.single_source_shortest_path_length(G.to_undirected(), origen)

        for nodo in nodos:
            niveles[nodo] = longitudes.get(nodo, 0)

        grupos = {}
        for nodo, nivel in niveles.items():
            grupos.setdefault(nivel, []).append(nodo)

        pos = {}
        for nivel, grupo in grupos.items():
            grupo = sorted(grupo)
            cantidad = len(grupo)

            for i, nodo in enumerate(grupo):
                y = (cantidad - 1) / 2 - i
                pos[nodo] = (nivel * 2.2, y * 1.5)

        return pos

    except Exception:
        return nx.spring_layout(G, seed=10, k=1.3)


def dibujar_grafo(G, ruta=None, origen=None, destino=None, dirigido=True):
    if ruta is None:
        ruta = []

    pos = posiciones_grafo(G, origen, destino)

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    aristas_ruta = []
    if ruta and len(ruta) > 1:
        for i in range(len(ruta) - 1):
            aristas_ruta.append((ruta[i], ruta[i + 1]))

    colores_nodos = []
    tamanos_nodos = []

    for nodo in G.nodes():
        if nodo == origen:
            colores_nodos.append("#2E86DE")
            tamanos_nodos.append(1800)
        elif nodo == destino:
            colores_nodos.append("#E74C3C")
            tamanos_nodos.append(1800)
        elif nodo in ruta:
            colores_nodos.append("#58D68D")
            tamanos_nodos.append(1650)
        else:
            colores_nodos.append("#F4F6F7")
            tamanos_nodos.append(1450)

    # Aristas normales
    aristas_normales = [e for e in G.edges() if e not in aristas_ruta]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=aristas_normales,
        arrows=dirigido,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        edge_color="#AAB7B8",
        connectionstyle="arc3,rad=0.03",
        ax=ax,
    )

    # Aristas de la ruta
    if aristas_ruta:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=aristas_ruta,
            arrows=dirigido,
            arrowstyle="-|>",
            arrowsize=24,
            width=4.2,
            edge_color="#27AE60",
            connectionstyle="arc3,rad=0.03",
            ax=ax,
        )

    # Nodos
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colores_nodos,
        node_size=tamanos_nodos,
        edgecolors="#2C3E50",
        linewidths=2,
        ax=ax,
    )

    # Etiquetas de nodos
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=13,
        font_weight="bold",
        font_color="#1B2631",
        ax=ax,
    )

    # Etiquetas de costos
    etiquetas_costos = {}
    for u, v, datos in G.edges(data=True):
        costo = datos.get("weight", "")
        etiquetas_costos[(u, v)] = formatear_numero(costo)

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=etiquetas_costos,
        font_size=11,
        font_color="#1B2631",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#D5D8DC",
            "alpha": 0.95,
        },
        label_pos=0.5,
        rotate=False,
        ax=ax,
    )

    ax.set_title(
        "Grafo de rutas y costos",
        fontsize=17,
        fontweight="bold",
        color="#1B2631",
        pad=18,
    )

    ax.text(
        0.01,
        -0.06,
        "Azul: origen   Rojo: destino   Verde: ruta más corta",
        transform=ax.transAxes,
        fontsize=11,
        color="#34495E",
    )

    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# INICIALIZACIÓN
# ============================================================
if "edges_df" not in st.session_state:
    st.session_state.edges_df = datos_ejercicio_imagen()

if "origen_input" not in st.session_state:
    st.session_state.origen_input = "A"

if "destino_input" not in st.session_state:
    st.session_state.destino_input = "J"

if "dirigido_input" not in st.session_state:
    st.session_state.dirigido_input = True


# ============================================================
# PANEL LATERAL
# ============================================================
with st.sidebar:
    st.header("Configuración")

    st.markdown("Carga rápida del ejercicio de la imagen:")

    if st.button("Cargar ejercicio A → J"):
        st.session_state.edges_df = datos_ejercicio_imagen()
        st.session_state.origen_input = "A"
        st.session_state.destino_input = "J"
        st.session_state.dirigido_input = True
        st.rerun()

    st.markdown("---")

    st.markdown("También puedes modificar la tabla principal para resolver otro problema.")


# ============================================================
# ENTRADA DEL PROBLEMA
# ============================================================
st.header("1. Ingreso de datos")

col_origen, col_destino, col_tipo = st.columns([1, 1, 1])

with col_origen:
    origen = st.text_input(
        "Nodo origen",
        key="origen_input"
    )
    origen = limpiar_nodo(origen)

with col_destino:
    destino = st.text_input(
        "Nodo destino",
        key="destino_input"
    )
    destino = limpiar_nodo(destino)

with col_tipo:
    dirigido = st.checkbox(
        "Grafo dirigido",
        key="dirigido_input"
    )

st.markdown("### Caminos del grafo")

st.caption(
    "Ingresa cada camino en una fila. Puedes agregar más filas desde la tabla."
)

edges_editados = st.data_editor(
    st.session_state.edges_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Origen": st.column_config.TextColumn(
            "Origen",
            help="Nodo desde donde sale el camino",
            required=True,
        ),
        "Destino": st.column_config.TextColumn(
            "Destino",
            help="Nodo al que llega el camino",
            required=True,
        ),
        "Costo": st.column_config.NumberColumn(
            "Costo",
            help="Costo, distancia o tiempo del camino",
            min_value=0.0,
            step=1.0,
            required=True,
        ),
    },
    hide_index=True,
)

st.session_state.edges_df = edges_editados


# ============================================================
# PROCESAMIENTO DE DATOS
# ============================================================
aristas, errores = procesar_aristas(edges_editados)

if aristas:
    nodos = obtener_nodos(aristas, origen, destino)
else:
    nodos = []

st.header("2. Resumen del problema")

col_resumen1, col_resumen2, col_resumen3 = st.columns(3)

with col_resumen1:
    st.metric("Nodos", len(nodos))

with col_resumen2:
    st.metric("Caminos", len(aristas))

with col_resumen3:
    st.metric("Tipo", "Dirigido" if dirigido else "No dirigido")

if errores:
    st.warning("Hay datos por revisar:")
    for error in errores:
        st.write("- " + error)

if aristas:
    df_caminos = pd.DataFrame(aristas, columns=["Origen", "Destino", "Costo"])
    st.dataframe(df_caminos, use_container_width=True, hide_index=True)


# ============================================================
# EJECUCIÓN
# ============================================================
st.header("3. Resolver")

if st.button("Calcular ruta más corta", type="primary"):
    if errores:
        st.error("Corrige los errores antes de calcular.")
    elif origen == "":
        st.error("Debes ingresar un nodo origen.")
    elif destino == "":
        st.error("Debes ingresar un nodo destino.")
    elif origen not in nodos:
        st.error(f"El nodo origen '{origen}' no aparece en los caminos ingresados.")
    elif destino not in nodos:
        st.error(f"El nodo destino '{destino}' no aparece en los caminos ingresados.")
    else:
        G = construir_grafo(aristas, dirigido=dirigido)

        ruta, costo, tablas, historial, distancia, predecesor = dijkstra_con_tablas(
            aristas=aristas,
            origen=origen,
            destino=destino,
            dirigido=dirigido,
        )

        st.header("4. Solución final")

        if ruta is None:
            st.error(f"No existe una ruta desde {origen} hasta {destino}.")
            st.markdown("### Grafo ingresado")
            dibujar_grafo(G, ruta=[], origen=origen, destino=destino, dirigido=dirigido)
        else:
            col_sol1, col_sol2 = st.columns([2, 1])

            with col_sol1:
                st.success(f"Ruta más corta: {' → '.join(ruta)}")

            with col_sol2:
                st.success(f"Costo mínimo: {formatear_numero(costo)}")

            st.markdown("### Interpretación")
            st.write(
                f"Para llegar desde **{origen}** hasta **{destino}**, "
                f"la ruta óptima es **{' → '.join(ruta)}**. "
                f"El costo total mínimo es **{formatear_numero(costo)}**."
            )

            st.markdown("### Detalle de la ruta elegida")

            detalle_ruta = []
            costo_acumulado = 0

            for i in range(len(ruta) - 1):
                u = ruta[i]
                v = ruta[i + 1]
                costo_tramo = G[u][v]["weight"]
                costo_acumulado += costo_tramo

                detalle_ruta.append(
                    {
                        "Paso": i + 1,
                        "Desde": u,
                        "Hasta": v,
                        "Costo del tramo": formatear_numero(costo_tramo),
                        "Costo acumulado": formatear_numero(costo_acumulado),
                    }
                )

            st.dataframe(
                pd.DataFrame(detalle_ruta),
                use_container_width=True,
                hide_index=True,
            )

            st.header("5. Grafo con la ruta resaltada")
            dibujar_grafo(G, ruta=ruta, origen=origen, destino=destino, dirigido=dirigido)

            st.header("6. Historial del algoritmo")
            st.code("\n".join(historial), language="text")

            st.header("7. Tablas iterativas")

            for item in tablas:
                st.markdown(
                    f"#### Iteración {item['iteracion']} - Nodo seleccionado: {item['nodo_actual']}"
                )
                st.dataframe(
                    item["tabla"],
                    use_container_width=True,
                    hide_index=True,
                )

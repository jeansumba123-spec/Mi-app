from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import sympy as sp
from scipy.optimize import minimize
from scipy.stats import chi2_contingency

try:
    from docx import Document
except Exception:  # pragma: no cover
    Document = None

st.set_page_config(
    page_title="Programación no lineal y análisis de datos",
    page_icon="📈",
    layout="wide",
)

PALETTE = {
    "primary": "#1A548C",
    "primary_dark": "#0F304F",
    "accent": "#008C7A",
    "soft": "#F0F7FA",
    "danger": "#BF332E",
    "success": "#338C40",
    "warning": "#ED9E1F",
}

VARIABLE_CATALOG = {
    "Demográficas": [
        ("Edad", ["edad"]),
        ("Género al nacer", ["sexo", "genero", "género"]),
        ("Ingresos", ["ingreso", "renta"]),
        ("Nivel de instrucción", ["nivel de instruccion", "nivel de instrucción", "instruccion", "instrucción"]),
        ("Zona de residencia", ["zona de residencia", "residencia"]),
    ],
    "Uso": [
        ("Estación", ["estacion", "estación"]),
        ("Zona de destino", ["zona de destino", "destino"]),
        ("Motivo de viaje", ["motivo de viaje", "motivo"]),
        ("Frecuencia de uso", ["frecuencia", "utiliza"]),
        ("Transporte anterior", ["transporte anterior"]),
        ("Bus anterior", ["bus anterior"]),
    ],
    "Percepción": [
        ("Satisfacción", ["satisfaccion", "satisfacción", "grado de satisfaccion", "grado de satisfacción"]),
        ("Factor de cambio", ["factor de cambio", "factor"]),
        ("Sugerencia", ["sugerencia", "observacion", "observación"]),
    ],
}

BIVARIABLE_CATALOG = [
    ("Género vs Satisfacción", ["sexo", "genero", "género"], ["satisfaccion", "satisfacción"]),
    ("Edad vs Frecuencia de uso", ["edad"], ["frecuencia", "utiliza"]),
    ("Ingresos vs Transporte anterior", ["ingreso", "renta"], ["transporte anterior"]),
    ("Motivo de viaje vs Frecuencia de uso", ["motivo de viaje", "motivo"], ["frecuencia", "utiliza"]),
    ("Nivel de instrucción vs Satisfacción", ["nivel de instruccion", "nivel de instrucción", "instruccion", "instrucción"], ["satisfaccion", "satisfacción"]),
    ("Edad vs Factor de cambio", ["edad"], ["factor de cambio", "factor"]),
    ("Ingresos vs Factor de cambio", ["ingreso", "renta"], ["factor de cambio", "factor"]),
    ("Estación vs Motivo de viaje", ["estacion", "estación"], ["motivo de viaje", "motivo"]),
]


def init_state() -> None:
    defaults = {
        "workbooks": {},
        "active_file": None,
        "history": [],
        "records": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_history(text: str, record: Optional[dict] = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(f"{timestamp} | {text}")
    if record is not None:
        record["timestamp"] = timestamp
        st.session_state.records.append(record)


def normalize_name(value: str) -> str:
    return str(value).strip().lower()


def find_column(data: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    columns = list(data.columns)
    for pattern in patterns:
        p = normalize_name(pattern)
        for col in columns:
            if p in normalize_name(col):
                return col
    return None


def clean_series(series: pd.Series) -> pd.Series:
    cleaned = series.dropna()
    if cleaned.dtype == object:
        cleaned = cleaned.astype(str).str.strip()
        cleaned = cleaned[cleaned != ""]
    return cleaned


def as_labels(series: pd.Series) -> pd.Series:
    return clean_series(series).astype(str).str.strip()


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean_series(series), errors="coerce").dropna()


def frequency_table(series: pd.Series) -> pd.DataFrame:
    labels = as_labels(series)
    counts = labels.value_counts(dropna=True, sort=False)
    result = pd.DataFrame({"Categoría": counts.index.astype(str), "Frecuencia": counts.values})
    total = result["Frecuencia"].sum()
    result["Porcentaje"] = np.where(total > 0, result["Frecuencia"] / total * 100, 0).round(2)
    return result


def load_workbook(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Usa CSV, XLSX o XLS.")


def sidebar() -> str:
    st.sidebar.title("📌 Menú")
    uploaded = st.sidebar.file_uploader(
        "Cargar archivos Excel o CSV",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )
    if uploaded:
        for file in uploaded:
            try:
                st.session_state.workbooks[file.name] = load_workbook(file)
                st.session_state.active_file = file.name
            except Exception as exc:
                st.sidebar.error(f"No se pudo cargar {file.name}: {exc}")

    if st.session_state.workbooks:
        names = list(st.session_state.workbooks.keys())
        current_index = names.index(st.session_state.active_file) if st.session_state.active_file in names else 0
        st.session_state.active_file = st.sidebar.selectbox("Archivo activo", names, index=current_index)
    else:
        st.sidebar.info("Aún no hay archivos cargados.")

    total_rows = sum(len(df) for df in st.session_state.workbooks.values())
    active_cols = 0
    if st.session_state.active_file:
        active_cols = st.session_state.workbooks[st.session_state.active_file].shape[1]
    st.sidebar.metric("Archivos", len(st.session_state.workbooks))
    st.sidebar.metric("Registros totales", total_rows)
    st.sidebar.metric("Variables activas", active_cols)

    return st.sidebar.radio(
        "Vista",
        ["Inicio", "Programación no lineal", "Análisis univariable", "Análisis bivariable", "Historial y exportación", "Ayuda"],
    )


def active_data() -> Optional[pd.DataFrame]:
    name = st.session_state.active_file
    if not name:
        return None
    return st.session_state.workbooks.get(name)


def home_view() -> None:
    st.title("📈 App de programación no lineal y análisis de datos")
    st.write(
        "Esta app transforma la lógica del proyecto MATLAB/AppDesigner a Streamlit y agrega un módulo para resolver problemas de programación no lineal."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("Carga Excel/CSV")
    c2.info("Optimiza funciones con restricciones")
    c3.info("Exporta historial y resultados")
    if active_data() is not None:
        st.subheader(f"Vista previa: {st.session_state.active_file}")
        st.dataframe(active_data().head(30), use_container_width=True)


@dataclass
class ParsedProblem:
    variables: List[str]
    objective: Callable[[np.ndarray], float]
    objective_sympy: sp.Expr
    constraints: List[dict]
    bounds: List[Tuple[Optional[float], Optional[float]]]


def parse_variables(text: str) -> List[str]:
    variables = [x.strip() for x in text.split(",") if x.strip()]
    if not variables:
        raise ValueError("Debes escribir al menos una variable, por ejemplo: x,y")
    if len(set(variables)) != len(variables):
        raise ValueError("Las variables no deben repetirse.")
    return variables


def parse_bounds(text: str, variables: List[str]) -> List[Tuple[Optional[float], Optional[float]]]:
    if not text.strip():
        return [(None, None) for _ in variables]
    result: Dict[str, Tuple[Optional[float], Optional[float]]] = {var: (None, None) for var in variables}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Límite inválido: {line}. Usa formato x:0,10")
        var, bounds = [part.strip() for part in line.split(":", 1)]
        if var not in variables:
            raise ValueError(f"La variable {var} no está en la lista de variables.")
        pieces = [p.strip() for p in bounds.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Límite inválido para {var}. Usa x:min,max")
        low = None if pieces[0] in ("", "None", "none", "-inf") else float(pieces[0])
        high = None if pieces[1] in ("", "None", "none", "inf") else float(pieces[1])
        result[var] = (low, high)
    return [result[var] for var in variables]


def make_numeric_function(expr: sp.Expr, symbols: List[sp.Symbol]) -> Callable[[np.ndarray], float]:
    func = sp.lambdify(symbols, expr, modules=["numpy"])

    def wrapped(values: np.ndarray) -> float:
        try:
            val = func(*values)
            return float(np.asarray(val, dtype=float))
        except Exception:
            return np.inf

    return wrapped


def parse_constraint(line: str, symbols_map: Dict[str, sp.Symbol], symbols: List[sp.Symbol]) -> dict:
    clean = line.strip()
    if not clean:
        raise ValueError("Restricción vacía.")
    if "<=" in clean:
        left, right = clean.split("<=", 1)
        expr = sp.sympify(right, locals=symbols_map) - sp.sympify(left, locals=symbols_map)
        kind = "ineq"
    elif ">=" in clean:
        left, right = clean.split(">=", 1)
        expr = sp.sympify(left, locals=symbols_map) - sp.sympify(right, locals=symbols_map)
        kind = "ineq"
    elif "=" in clean:
        left, right = clean.split("=", 1)
        expr = sp.sympify(left, locals=symbols_map) - sp.sympify(right, locals=symbols_map)
        kind = "eq"
    else:
        expr = sp.sympify(clean, locals=symbols_map)
        kind = "ineq"
    func = make_numeric_function(expr, symbols)
    return {"type": kind, "fun": func, "text": clean, "expr": expr}


def parse_problem(objective_text: str, variable_text: str, x0_text: str, bounds_text: str, constraints_text: str, sense: str) -> Tuple[ParsedProblem, np.ndarray]:
    variables = parse_variables(variable_text)
    symbols = sp.symbols(variables)
    if len(variables) == 1:
        symbols = [symbols]
    else:
        symbols = list(symbols)
    symbols_map = {name: symbol for name, symbol in zip(variables, symbols)}
    objective_expr = sp.sympify(objective_text, locals=symbols_map)
    objective = make_numeric_function(objective_expr, symbols)
    if sense == "Maximizar":
        original_objective = objective
        objective = lambda values: -original_objective(values)
    x0 = np.array([float(x.strip()) for x in x0_text.split(",") if x.strip()], dtype=float)
    if len(x0) != len(variables):
        raise ValueError("El punto inicial debe tener el mismo número de valores que variables.")
    bounds = parse_bounds(bounds_text, variables)
    constraints = []
    for raw in constraints_text.splitlines():
        if raw.strip():
            constraints.append(parse_constraint(raw, symbols_map, symbols))
    return ParsedProblem(variables, objective, objective_expr, constraints, bounds), x0


def optimization_view() -> None:
    st.title("🧮 Programación no lineal")
    st.caption("Resuelve problemas de minimización o maximización con SLSQP de SciPy.")

    left, right = st.columns([1, 1])
    with left:
        sense = st.selectbox("Tipo de problema", ["Minimizar", "Maximizar"])
        objective = st.text_input("Función objetivo f(x)", "x**2 + y**2")
        variables = st.text_input("Variables separadas por coma", "x,y")
        x0 = st.text_input("Punto inicial", "1,1")
        bounds = st.text_area("Límites opcionales", "x:0,10\ny:0,10", help="Formato: variable:min,max. Usa vacío para sin límite.")
        constraints = st.text_area("Restricciones", "x + y >= 5", help="Una por línea. Acepta <=, >= o =.")
        method = st.selectbox("Método", ["SLSQP", "trust-constr"])
        run = st.button("Resolver", type="primary")

    with right:
        st.markdown("**Ejemplo rápido**")
        st.code("""Función: x**2 + y**2
Variables: x,y
Inicial: 1,1
Límites:
x:0,10
y:0,10
Restricción:
x + y >= 5""", language="text")
        st.warning("Para multiplicar usa *, por ejemplo 2*x. Para potencias usa **, por ejemplo x**2.")

    if run:
        try:
            problem, x0_values = parse_problem(objective, variables, x0, bounds, constraints, sense)
            scipy_constraints = [{"type": c["type"], "fun": c["fun"]} for c in problem.constraints]
            result = minimize(
                problem.objective,
                x0_values,
                method=method,
                bounds=problem.bounds,
                constraints=scipy_constraints,
                options={"maxiter": 1000, "disp": False},
            )
            final_value = float(result.fun)
            if sense == "Maximizar":
                final_value = -final_value
            solution = pd.DataFrame({"Variable": problem.variables, "Valor óptimo": np.round(result.x, 8)})
            st.subheader("Resultado")
            if result.success:
                st.success("Optimización completada correctamente.")
            else:
                st.warning(f"El método terminó con aviso: {result.message}")
            c1, c2 = st.columns(2)
            c1.metric("Valor óptimo", f"{final_value:.8g}")
            c2.metric("Iteraciones", int(getattr(result, "nit", 0)))
            st.dataframe(solution, use_container_width=True)
            st.markdown("**Restricciones evaluadas en la solución**")
            rows = []
            for c in problem.constraints:
                rows.append({"Restricción": c["text"], "Tipo SciPy": c["type"], "Valor interno": c["fun"](result.x)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No se ingresaron restricciones.")
            add_history(
                f"Optimización | {sense} | f={objective} | valor={final_value:.6g}",
                {
                    "type": "Optimización",
                    "label": objective,
                    "table": solution,
                    "stats": f"{sense}: {objective}\nValor óptimo: {final_value}\nMensaje: {result.message}",
                },
            )

            if len(problem.variables) == 2:
                st.subheader("Visualización 2D")
                x_name, y_name = problem.variables
                x_low, x_high = problem.bounds[0]
                y_low, y_high = problem.bounds[1]
                x_low = -10 if x_low is None else x_low
                x_high = 10 if x_high is None else x_high
                y_low = -10 if y_low is None else y_low
                y_high = 10 if y_high is None else y_high
                xs = np.linspace(x_low, x_high, 80)
                ys = np.linspace(y_low, y_high, 80)
                xx, yy = np.meshgrid(xs, ys)
                z = np.zeros_like(xx, dtype=float)
                raw_func = make_numeric_function(problem.objective_sympy, [sp.Symbol(x_name), sp.Symbol(y_name)])
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        z[i, j] = raw_func(np.array([xx[i, j], yy[i, j]]))
                fig = px.contour(x=xs, y=ys, z=z, labels={"x": x_name, "y": y_name, "color": "f"})
                fig.add_scatter(x=[result.x[0]], y=[result.x[1]], mode="markers+text", text=["Óptimo"], textposition="top center")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo resolver el problema: {exc}")


def univariable_view() -> None:
    st.title("📊 Análisis univariable")
    data = active_data()
    if data is None:
        st.info("Carga un archivo Excel o CSV desde el menú lateral.")
        return
    st.caption(f"Archivo activo: {st.session_state.active_file}")
    mode = st.radio("Selección de variable", ["Catálogo automático", "Columna manual"], horizontal=True)
    if mode == "Catálogo automático":
        group = st.selectbox("Grupo", list(VARIABLE_CATALOG.keys()))
        label = st.selectbox("Variable", [x[0] for x in VARIABLE_CATALOG[group]])
        patterns = dict(VARIABLE_CATALOG[group])[label]
        column = find_column(data, patterns)
        if column is None:
            st.error(f"No se encontró una columna compatible con: {label}")
            return
    else:
        column = st.selectbox("Columna", data.columns)
        label = column
    chart_type = st.selectbox("Tipo de gráfico", ["Barras", "Pastel", "Pareto"])
    table = frequency_table(data[column])
    numeric = as_numeric(data[column])
    st.subheader(label)
    c1, c2, c3 = st.columns(3)
    c1.metric("N válido", int(table["Frecuencia"].sum()))
    if not table.empty:
        mode_row = table.loc[table["Frecuencia"].idxmax()]
        c2.metric("Moda", str(mode_row["Categoría"]))
        c3.metric("% moda", f"{mode_row['Porcentaje']:.1f}%")
    if len(numeric) > 0:
        st.write(f"Media: **{numeric.mean():.2f}** | Mediana: **{numeric.median():.2f}** | Desviación estándar: **{numeric.std():.2f}**")
    st.dataframe(table, use_container_width=True)
    if chart_type == "Barras":
        fig = px.bar(table, x="Categoría", y="Frecuencia", title=label)
    elif chart_type == "Pastel":
        fig = px.pie(table, names="Categoría", values="Frecuencia", title=label)
    else:
        pareto = table.sort_values("Frecuencia", ascending=False)
        fig = px.bar(pareto, x="Categoría", y="Frecuencia", title=f"Pareto: {label}")
    st.plotly_chart(fig, use_container_width=True)
    if not table.empty:
        st.success(f"La categoría con mayor presencia en {label} es {mode_row['Categoría']}, con {mode_row['Porcentaje']:.1f}% de los registros válidos.")
    if st.button("Guardar este análisis en historial"):
        add_history(
            f"Univariable | {label} | {st.session_state.active_file}",
            {"type": "Univariable", "label": label, "table": table, "stats": f"N válido: {int(table['Frecuencia'].sum())}"},
        )
        st.toast("Análisis guardado.")


def bivariable_view() -> None:
    st.title("▦ Análisis bivariable")
    data = active_data()
    if data is None:
        st.info("Carga un archivo Excel o CSV desde el menú lateral.")
        return
    st.caption(f"Archivo activo: {st.session_state.active_file}")
    mode = st.radio("Selección", ["Catálogo automático", "Columnas manuales"], horizontal=True)
    if mode == "Catálogo automático":
        pair_label = st.selectbox("Cruce", [x[0] for x in BIVARIABLE_CATALOG])
        selected = next(x for x in BIVARIABLE_CATALOG if x[0] == pair_label)
        left_col = find_column(data, selected[1])
        right_col = find_column(data, selected[2])
        if left_col is None or right_col is None:
            st.error("No se encontraron una o ambas columnas para este cruce.")
            return
    else:
        left_col = st.selectbox("Variable fila", data.columns)
        right_col = st.selectbox("Variable columna", data.columns)
        pair_label = f"{left_col} vs {right_col}"
    chart_type = st.selectbox("Tipo de gráfico", ["Heatmap", "Barras agrupadas", "Barras apiladas"])
    temp = pd.DataFrame({"Fila": as_labels(data[left_col]), "Columna": as_labels(data[right_col])}).dropna()
    table = pd.crosstab(temp["Fila"], temp["Columna"])
    if table.empty:
        st.warning("No hay pares válidos para analizar.")
        return
    chi2, p_value, dof, expected = chi2_contingency(table)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-cuadrado", f"{chi2:.3f}")
    c2.metric("gl", int(dof))
    c3.metric("p-valor", f"{p_value:.4f}")
    if p_value < 0.05:
        st.success("Asociación significativa al 5%.")
    elif p_value < 0.10:
        st.warning("Señal moderada, no concluyente al 5%.")
    else:
        st.error("No hay evidencia suficiente de asociación.")
    st.dataframe(table, use_container_width=True)
    if chart_type == "Heatmap":
        fig = px.imshow(table, text_auto=True, aspect="auto", title=pair_label)
    else:
        plot_data = table.reset_index().melt(id_vars="Fila", var_name="Columna", value_name="Frecuencia")
        barmode = "group" if chart_type == "Barras agrupadas" else "stack"
        fig = px.bar(plot_data, x="Fila", y="Frecuencia", color="Columna", barmode=barmode, title=pair_label)
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Guardar este cruce en historial"):
        add_history(
            f"Bivariable | {pair_label} | {st.session_state.active_file} | p={p_value:.4f}",
            {"type": "Bivariable", "label": pair_label, "table": table.reset_index(), "stats": f"Chi2={chi2:.3f}; gl={dof}; p={p_value:.4f}"},
        )
        st.toast("Cruce guardado.")


def build_excel_report() -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary = pd.DataFrame({"Historial": st.session_state.history})
        summary.to_excel(writer, index=False, sheet_name="Historial")
        for i, record in enumerate(st.session_state.records, start=1):
            sheet = f"{i:02d}_{record['type']}"[:31]
            record["table"].to_excel(writer, index=False, sheet_name=sheet)
    return output.getvalue()


def build_docx_report() -> Optional[bytes]:
    if Document is None:
        return None
    document = Document()
    document.add_heading("Reporte de análisis", level=1)
    document.add_paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    document.add_heading("Historial", level=2)
    for item in st.session_state.history:
        document.add_paragraph(item)
    for record in st.session_state.records:
        document.add_heading(f"{record['type']}: {record['label']}", level=2)
        document.add_paragraph(record.get("stats", ""))
        table_df = record["table"].reset_index(drop=True)
        doc_table = document.add_table(rows=1, cols=len(table_df.columns))
        for j, col in enumerate(table_df.columns):
            doc_table.rows[0].cells[j].text = str(col)
        for _, row in table_df.head(80).iterrows():
            cells = doc_table.add_row().cells
            for j, value in enumerate(row):
                cells[j].text = str(value)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def history_view() -> None:
    st.title("🕘 Historial y exportación")
    if not st.session_state.history:
        st.info("Aún no hay análisis guardados en esta sesión.")
        return
    st.text_area("Historial", "\n".join(st.session_state.history), height=260)
    col1, col2 = st.columns(2)
    col1.download_button("Descargar reporte Excel", build_excel_report(), "reporte_analisis.xlsx")
    docx = build_docx_report()
    if docx is not None:
        col2.download_button("Descargar reporte Word", docx, "reporte_analisis.docx")
    if st.button("Limpiar historial"):
        st.session_state.history = []
        st.session_state.records = []
        st.rerun()


def help_view() -> None:
    st.title("📘 Ayuda")
    st.markdown(
        """
### Cómo correr la app
1. Instala dependencias: `pip install -r requirements.txt`
2. Ejecuta: `streamlit run app.py`
3. Abre la dirección local que muestra Streamlit.

### Sintaxis para programación no lineal
- Multiplicación: `2*x`
- Potencia: `x**2`
- Raíz: `sqrt(x)`
- Funciones: `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`
- Restricciones: `x + y >= 5`, `x <= 10`, `x + 2*y = 8`

### Archivos de datos
Puedes cargar `.xlsx`, `.xls` o `.csv`. La app detecta columnas usando palabras clave similares a las usadas en el código MATLAB original.
        """
    )


def main() -> None:
    init_state()
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PALETTE['soft']}; }}
        section[data-testid="stSidebar"] {{ background-color: {PALETTE['primary_dark']}; }}
        section[data-testid="stSidebar"] * {{ color: white; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    view = sidebar()
    if view == "Inicio":
        home_view()
    elif view == "Programación no lineal":
        optimization_view()
    elif view == "Análisis univariable":
        univariable_view()
    elif view == "Análisis bivariable":
        bivariable_view()
    elif view == "Historial y exportación":
        history_view()
    else:
        help_view()


if __name__ == "__main__":
    main()

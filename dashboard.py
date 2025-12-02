import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Dashboard Completo de Gastos", layout="wide")


# --- 1. FUNCIONES DE CLASIFICACIÓN (Reglas de Negocio) ---
def clasificar_tercero(nombre):
    if not isinstance(nombre, str):
        return "Otros"
    nombre = nombre.upper()
    if any(x in nombre for x in ["S.L", "SLU", "S.L.", "S.L.U", "SL", "S.L.U."]):
        return "Sociedad Limitada"
    elif any(x in nombre for x in ["S.A", "S.A.", "S.A.U.", "SA", "S.A.U ", "SAU"]):
        return "Sociedad Anonima"
    elif any(x in nombre for x in ["S.C", "S.C.P", "S.C.P.", "S.C.", "SCP"]):
        return "Sociedad Civil"
    elif any(
        x in nombre for x in ["UNIVERSIDAD", "UPV", "POLI", "UNI", "ESCUELA", "CENTRO"]
    ):
        return "Universidad"
    else:
        return "Otros"


def clasificar_concepto(concepto):
    if not isinstance(concepto, str):
        return "Otros"
    concepto = concepto.upper()
    if "OFICINA" in concepto:
        return "Material de oficina"
    elif "LABORATORIO" in concepto:
        return "Material de laboratorio"
    elif "MANTENIMIENTO" in concepto:
        return "Mantenimiento"
    elif "SERVICIO" in concepto:
        return "Servicios"
    elif "INFORMÁTIC" in concepto or "SOFTWARE" in concepto:
        return "Informática"
    elif "TRABAJO" in concepto:
        return "Trabajos"
    else:
        return "Otros"


def clasificar_centro(nombre):
    if not isinstance(nombre, str):
        return "Otros"
    nombre = nombre.upper()
    if any(x in nombre for x in ["DEP.", "DEPARTAMENTO"]):
        return "Departamento"
    elif any(x in nombre for x in ["INSTITUTO", "INST."]):
        return "Instituto"
    elif "SERV" in nombre:
        return "Servicio"
    elif any(
        x in nombre for x in ["ESC", "ESCUELA", "FACULTAD", "EPS", "E.P.S.", "ETS"]
    ):
        return "Centro Docente"
    elif any(x in nombre for x in ["GESTIÓN", "DIRECCIÓN"]):
        return "Gestión/Admin"
    else:
        return "Otros"


# --- 2. CARGA DE DATOS ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])


@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        # Normalizar columnas a mayúsculas
        df.columns = [c.strip().upper() for c in df.columns]

        # Limpieza de nulos y espacios
        for col in ["TERCERO", "CONCEPTO ECONÓMICO", "CENTRO DIRECTIVO"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
            else:
                df[col] = "DESCONOCIDO"

        # Aplicar clasificaciones
        df["CATEGORIA_TERCERO"] = df["TERCERO"].apply(clasificar_tercero)
        df["CATEGORIA_CONCEPTO"] = df["CONCEPTO ECONÓMICO"].apply(clasificar_concepto)
        df["TIPO_CENTRO"] = df["CENTRO DIRECTIVO"].apply(clasificar_centro)
    else:
        # Datos de prueba (Dummies) para demostración
        data = []
        years = [2022, 2023, 2024]
        terceros = [
            "EMPRESA A S.L.",
            "EMPRESA B S.A.",
            "JUAN PEREZ",
            "UNIVERSIDAD X",
            "TALLERES S.C.",
        ]
        conceptos = [
            "COMPRA MATERIAL OFICINA",
            "SUMINISTRO LABORATORIO",
            "MANTENIMIENTO",
            "TRABAJOS VARIOS",
            "LICENCIA SOFTWARE",
            "LIBROS",
            "GASES INDUSTRIALES",
            "LIMPIEZA",
            "SEGURIDAD",
            "AUDITORIA",
            "ALQUILER",
            "HARDWARE",
            "CONSULTORIA",
        ]
        centros = [
            "DEP. INFORMATICA",
            "INSTITUTO FISICA",
            "GESTIÓN INVESTIGACIÓN",
            "ESCUELA AGRONOMOS",
        ]

        for _ in range(3000):
            data.append(
                [
                    np.random.choice(years),
                    np.random.choice(terceros),
                    np.random.uniform(50, 5000),
                    np.random.choice(conceptos),
                    np.random.choice(centros),
                ]
            )
        df = pd.DataFrame(
            data,
            columns=[
                "YEAR",
                "TERCERO",
                "IMPORTE",
                "CONCEPTO ECONÓMICO",
                "CENTRO DIRECTIVO",
            ],
        )

        df["CATEGORIA_TERCERO"] = df["TERCERO"].apply(clasificar_tercero)
        df["CATEGORIA_CONCEPTO"] = df["CONCEPTO ECONÓMICO"].apply(clasificar_concepto)
        df["TIPO_CENTRO"] = df["CENTRO DIRECTIVO"].apply(clasificar_centro)
    return df


gastos = load_data(uploaded_file)

# --- 3. FILTROS GLOBALES ---
st.sidebar.header("2. Filtros Globales")
all_years = sorted(gastos["YEAR"].unique())
sel_years = st.sidebar.multiselect("Años", all_years, default=all_years)

# Filtrado inicial por años
df_main = gastos[gastos["YEAR"].isin(sel_years)].copy()

st.title("📊 Dashboard de Gastos: Análisis Integral")
st.markdown("---")

# --- 4. MOTORES DE GRÁFICOS ---


# A. MOTOR FLEXIBLE (El que ya tenías: Barras/Líneas/Pie de categorías)
def render_flexible_analysis(df, category_col, section_title):
    st.subheader(f"📌 Análisis General por {section_title}")

    col_controls, col_graph = st.columns([1, 3])

    with col_controls:
        st.markdown("**Configuración**")
        chart_type = st.radio(
            f"Tipo:", ["Barras", "Líneas", "Area", "Pie"], key=f"cht_{category_col}"
        )
        agg_func = st.selectbox(
            f"Métrica:",
            ["Suma Total (€)", "Promedio (€)", "Conteo"],
            key=f"agg_{category_col}",
        )

        # Filtro local
        unique_cats = sorted(df[category_col].unique())
        sel_cats = st.multiselect(
            f"Filtrar {section_title}:",
            unique_cats,
            default=unique_cats,
            key=f"fil_{category_col}",
        )

    df_local = df[df[category_col].isin(sel_cats)]

    if df_local.empty:
        st.info("Sin datos con los filtros actuales.")
        return

    # Agrupación
    if agg_func == "Suma Total (€)":
        df_grouped = (
            df_local.groupby(["YEAR", category_col])["IMPORTE"].sum().reset_index()
        )
        y_val = "IMPORTE"
    elif agg_func == "Promedio (€)":
        df_grouped = (
            df_local.groupby(["YEAR", category_col])["IMPORTE"].mean().reset_index()
        )
        y_val = "IMPORTE"
    else:
        df_grouped = (
            df_local.groupby(["YEAR", category_col])["IMPORTE"].count().reset_index()
        )
        y_val = "IMPORTE"

    with col_graph:
        if chart_type == "Pie":
            fig = px.pie(
                df_grouped,
                values=y_val,
                names=category_col,
                title=f"Distribución {agg_func}",
            )
        elif chart_type == "Barras":
            fig = px.bar(
                df_grouped,
                x="YEAR",
                y=y_val,
                color=category_col,
                barmode="group",
                title=f"{agg_func} por Año",
            )
        elif chart_type == "Líneas":
            fig = px.line(
                df_grouped,
                x="YEAR",
                y=y_val,
                color=category_col,
                markers=True,
                title=f"Tendencia {agg_func}",
            )
        elif chart_type == "Area":
            fig = px.area(
                df_grouped,
                x="YEAR",
                y=y_val,
                color=category_col,
                title=f"Acumulado {agg_func}",
            )

        st.plotly_chart(fig, use_container_width=True)


# B. MOTOR NOTEBOOK (Nuevo: Top N + Otros específico)
def render_notebook_style_analysis(df, raw_col, label):
    """
    Replica la lógica exacta del notebook:
    1. Calcula el Top N de la columna 'cruda' (ej. nombre real del proveedor).
    2. Agrupa el resto en 'Otros'.
    3. Muestra Gráfico de Evolución (Líneas) y Composición (Barras 100%).
    """
    with st.expander(
        f"🔎 Análisis Detallado Top N: {label} (Estilo Notebook)", expanded=False
    ):
        col_conf, col_viz = st.columns([1, 4])

        with col_conf:
            n_top = st.slider(f"Top {label} a mostrar:", 3, 20, 10, key=f"n_{raw_col}")

        # 1. Identificar Top N
        top_list = df.groupby(raw_col)["IMPORTE"].sum().nlargest(n_top).index.tolist()

        # 2. Crear columna agrupada
        df_calc = df.copy()
        col_grouped = f"{raw_col}_AGRUPADO"
        df_calc[col_grouped] = df_calc[raw_col].apply(
            lambda x: x if x in top_list else "Otros"
        )

        # 3. Agrupar
        df_agrupado = (
            df_calc.groupby(["YEAR", col_grouped])["IMPORTE"].sum().reset_index()
        )

        with col_viz:
            # Gráfico 1: Líneas (Evolución)
            fig_line = px.line(
                df_agrupado,
                x="YEAR",
                y="IMPORTE",
                color=col_grouped,
                markers=True,
                title=f"Evolución Anual: Top {n_top} {label} + Otros",
                labels={"IMPORTE": "Importe Total (€)"},
            )
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")

            # Gráfico 2: Barras 100% (Composición)
            # Calculamos porcentajes manualmente para el gráfico stack 100%
            df_total_year = (
                df_agrupado.groupby("YEAR")["IMPORTE"]
                .sum()
                .reset_index()
                .rename(columns={"IMPORTE": "TOTAL_ANUAL"})
            )
            df_merged = pd.merge(df_agrupado, df_total_year, on="YEAR")
            df_merged["PORCENTAJE"] = df_merged["IMPORTE"] / df_merged["TOTAL_ANUAL"]

            fig_bar = px.bar(
                df_merged,
                x="YEAR",
                y="PORCENTAJE",
                color=col_grouped,
                title=f"Composición del Gasto (Top {n_top} {label})",
                text_auto=".1%",
                labels={"PORCENTAJE": "% del Gasto Anual"},
            )
            fig_bar.update_layout(yaxis_tickformat=".0%")  # Formato porcentaje
            st.plotly_chart(fig_bar, use_container_width=True)


# --- 5. ESTRUCTURA DE PESTAÑAS ---

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏢 Proveedores (Terceros)",
        "📦 Conceptos Económicos",
        "🏛️ Centros Directivos",
        "📋 Datos Brutos",
    ]
)

# PESTAÑA 1: TERCEROS
with tab1:
    # 1. Flexible (Categoría S.A, S.L...)
    render_flexible_analysis(df_main, "CATEGORIA_TERCERO", "Tipo de Sociedad")
    # 2. Notebook (Top N Proveedores reales)
    render_notebook_style_analysis(df_main, "TERCERO", "Proveedores")

# PESTAÑA 2: CONCEPTOS
with tab2:
    # 1. Flexible (Categoría Oficina, Laboratorio...)
    render_flexible_analysis(df_main, "CATEGORIA_CONCEPTO", "Familia de Gasto")
    # 2. Notebook (Top N Conceptos reales)
    render_notebook_style_analysis(df_main, "CONCEPTO ECONÓMICO", "Conceptos")

# PESTAÑA 3: CENTROS
with tab3:
    # 1. Flexible (Tipo Departamento, Instituto...)
    render_flexible_analysis(df_main, "TIPO_CENTRO", "Tipo de Organismo")
    # 2. Notebook (Top N Centros reales)
    render_notebook_style_analysis(df_main, "CENTRO DIRECTIVO", "Centros")

# PESTAÑA 4: DATOS
with tab4:
    st.subheader("Explorador de Datos")
    st.dataframe(df_main, use_container_width=True)

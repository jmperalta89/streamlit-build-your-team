import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io

# ------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ------------------------------
st.set_page_config(page_title="Team Splitter", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        .footer {
            position: relative;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-top: 2em;
        }
        .card {
            background: #ffffff;
            color: #000000 !important;
            padding: 1em;
            border-radius: 1em;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 1em;
        }
        .modo-banner {
            padding: 0.5em;
            border-radius: 0.5em;
            font-weight: bold;
            margin-bottom: 1em;
            text-align: center;
        }
        .modo-express { background-color: #d1f7c4; color: #256029; }
        .modo-avanzado { background-color: #cce5ff; color: #084298; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Generador de Equipos Balanceados")
st.caption("**Build your Team APP**")

st.markdown("Subí un archivo `.xlsx` con tus jugadores o cargá los datos en la tabla editable. Elegí entre **modo Express** y **modo Avanzado** según el nivel de detalle.")

# ------------------------------
# OPCIÓN DE MODO
# ------------------------------
modo = st.radio("Seleccioná el modo:", ["Express", "Avanzado"], horizontal=True)

if modo == "Express":
    columnas_esperadas = ["NOMBRE", "SEXO", "POSICION", "PUNTAJE"]
    st.markdown('<div class="modo-banner modo-express">🟢 Estás en MODO EXPRESS</div>', unsafe_allow_html=True)
else:
    columnas_esperadas = [
        "NOMBRE",
        "POSICION",
        "EDAD",
        "RESISTENCIA",
        "VELOCIDAD",
        "DRIBLE",
        "PEGADA",
        "PASE",
        "DEFENSA",
        "SEXO",
    ]
    st.markdown('<div class="modo-banner modo-avanzado">🔵 Estás en MODO AVANZADO</div>', unsafe_allow_html=True)

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def generar_equipos(df, cabeza_a=None, cabeza_b=None):
    df_features = df.copy()

    # Normalización de SEXO
    if "SEXO" in df_features.columns:
        df_features["SEXO"] = (
            df_features["SEXO"]
            .astype(str)
            .str.upper()
            .map({"M": 1, "H": 1, "HOMBRE": 1, "F": 0, "MUJER": 0})
        )
        df_features["SEXO"] = df_features["SEXO"].fillna(0.5)

    # Selección de features
    features = [c for c in df_features.columns if c not in ["NOMBRE", "POSICION"]]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_features[features])

    # Clustering con KMeans en 2 grupos
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    df_features["cluster"] = kmeans.fit_predict(X)

    equipo_a = df[df_features["cluster"] == 0]
    equipo_b = df[df_features["cluster"] == 1]

    # Forzar cabezas de serie
    if cabeza_a and cabeza_a in df["NOMBRE"].values:
        jugador_a = df[df["NOMBRE"] == cabeza_a]
        equipo_a = pd.concat([equipo_a, jugador_a]).drop_duplicates(subset=["NOMBRE"])
        equipo_b = equipo_b[equipo_b["NOMBRE"] != cabeza_a]

    if cabeza_b and cabeza_b in df["NOMBRE"].values:
        jugador_b = df[df["NOMBRE"] == cabeza_b]
        equipo_b = pd.concat([equipo_b, jugador_b]).drop_duplicates(subset=["NOMBRE"])
        equipo_a = equipo_a[equipo_a["NOMBRE"] != cabeza_b]

    return equipo_a, equipo_b


def resumir_stats(equipo):
    cols_stats = [c for c in equipo.columns if c not in ["NOMBRE", "POSICION", "SEXO"]]
    return equipo[cols_stats].mean(numeric_only=True)


# ------------------------------
# CARGA DE DATOS
# ------------------------------
archivo = st.file_uploader("📂 Cargar archivo .xlsx", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)
else:
    # Crear tabla editable vacía
    default_df = pd.DataFrame(columns=columnas_esperadas)

    # Configuración dinámica de columnas
    column_config = {
        "SEXO": st.column_config.SelectboxColumn("Sexo", options=["M", "F"], required=True),
        "POSICION": st.column_config.SelectboxColumn("Posición", options=["ATAQUE", "DEFENSA", "ARQUERO"], required=True),
    }

    if "PUNTAJE" in columnas_esperadas:
        column_config["PUNTAJE"] = st.column_config.NumberColumn(
            "Puntaje",
            help="De 0 a 100. Rango: 90-100 = 'Crack', 70-89 = 'Muy bueno', 50-69 = 'Promedio', <50 = 'A mejorar'",
            min_value=0,
            max_value=100,
            step=1,
        )

    df = st.data_editor(
        default_df,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
    )

# ------------------------------
# GENERAR EQUIPOS
# ------------------------------
if all(col in df.columns for col in columnas_esperadas) and not df.empty:
    st.success("✅ Datos listos para generar equipos")

    col1, col2 = st.columns(2)
    with col1:
        cabeza_a = st.selectbox("Cabeza de grupo A (opcional)", options=[None] + df["NOMBRE"].dropna().tolist())
    with col2:
        cabeza_b = st.selectbox("Cabeza de grupo B (opcional)", options=[None] + df["NOMBRE"].dropna().tolist())

    if st.button("🎲 Generar Equipos", type="primary"):
        with st.spinner("Generando equipos balanceados..."):
            equipo_a, equipo_b = generar_equipos(df, cabeza_a, cabeza_b)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟥 Equipo A")
            for _, row in equipo_a.iterrows():
                st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>📌 {row['POSICION']}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 🟦 Equipo B")
            for _, row in equipo_b.iterrows():
                st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>📌 {row['POSICION']}</div>", unsafe_allow_html=True)

        # Radar chart comparativo (solo si Avanzado)
        stats_a = resumir_stats(equipo_a)
        stats_b = resumir_stats(equipo_b)

        if not stats_a.empty and not stats_b.empty:
            radar_df = pd.DataFrame(
                {"Habilidad": stats_a.index, "Equipo A": stats_a.values, "Equipo B": stats_b.values}
            )
            radar_df = pd.melt(radar_df, id_vars=["Habilidad"], var_name="Equipo", value_name="Valor")
            fig = px.line_polar(radar_df, r="Valor", theta="Habilidad", color="Equipo", line_close=True)
            st.plotly_chart(fig, use_container_width=True)

        # Gráfico de medias
        medias = pd.DataFrame({
            "Equipo": ["A"] * len(stats_a) + ["B"] * len(stats_b),
            "Variable": list(stats_a.index) + list(stats_b.index),
            "Media": list(stats_a.values) + list(stats_b.values),
        })
        fig_media = px.bar(medias, x="Variable", y="Media", color="Equipo", barmode="group", title="Promedio de variables por equipo")
        st.plotly_chart(fig_media, use_container_width=True)

        # Distribución de sexo
        sexo_dist = pd.concat([
            equipo_a.assign(Equipo="A"),
            equipo_b.assign(Equipo="B")
        ])
        fig_sexo = px.histogram(sexo_dist, x="SEXO", color="Equipo", barmode="group", title="Distribución de sexo por equipo")
        st.plotly_chart(fig_sexo, use_container_width=True)

        # Exportar a Excel
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                equipo_a.to_excel(writer, sheet_name="Equipo A", index=False)
                equipo_b.to_excel(writer, sheet_name="Equipo B", index=False)
        except ModuleNotFoundError:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                equipo_a.to_excel(writer, sheet_name="Equipo A", index=False)
                equipo_b.to_excel(writer, sheet_name="Equipo B", index=False)

        st.download_button(
            label="📥 Descargar equipos en Excel",
            data=output.getvalue(),
            file_name="equipos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Leyenda de puntajes en modo Express
    if "PUNTAJE" in df.columns:
        st.markdown("#### 🎚️ Escala de Puntajes")
        st.markdown("""
        - 🟩 **90-100**: Crack!  
        - 🟨 **70-89**: Muy bueno  
        - 🟧 **50-69**: Promedio  
        - 🟥 **0-49**: A mejorar  
        """)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        ☕ Si te gustó esta app y querés invitarme un cafecito:<br>
        <a href="https://cafecito.app/jmperalta" target="_blank">Cafecito</a>  
        <br><br>
        Contacto: <a href="https://www.linkedin.com/in/peraltajm/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

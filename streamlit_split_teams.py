import streamlit as st
import pandas as pd
import numpy as np
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
# OPCIONES: MODO y TIPO FUTBOL
# ------------------------------
modo = st.radio("Seleccioná el modo:", ["Express", "Avanzado"], horizontal=True)
tipo_futbol = st.selectbox("Tipo de fútbol:", ["Mixto", "Masculino", "Femenino"])

if modo == "Express":
    st.markdown('<div class="modo-banner modo-express">🟢 Estás en MODO EXPRESS</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="modo-banner modo-avanzado">🔵 Estás en MODO AVANZADO</div>', unsafe_allow_html=True)

# ------------------------------
# Columnas esperadas según modo
# ------------------------------
if modo == "Express":
    columnas_esperadas = ["NOMBRE", "POSICION", "PUNTAJE"]
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
    ]

# SEXO: solo si Mixto mostramos la columna SEXO como obligatoria en la edición
if tipo_futbol == "Mixto":
    columnas_esperadas = ["NOMBRE", "SEXO", *[c for c in columnas_esperadas if c != "NOMBRE"]]

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def normalizar_sexo(col):
    # Normaliza valores comunes
    m = {"M": "M", "H": "M", "HOMBRE": "M", "MAS": "M", "F": "F", "MUJER": "F", "W": "F"}
    return col.astype(str).str.upper().map(m).where(lambda x: x.isin(["M","F"]), other=np.nan)

def resumir_stats(equipo):
    cols_stats = [c for c in equipo.columns if c not in ["NOMBRE", "POSICION", "SEXO"]]
    if not cols_stats:
        return pd.Series(dtype=float)
    return equipo[cols_stats].mean(numeric_only=True)

def _asignacion_greedy_balance(poblacion_df, size_a, size_b, cabeza_a_name=None, cabeza_b_name=None, sexo_mode=False):
    """
    Algoritmo greedy:
    - Coloca cabezas en sus equipos.
    - Si sexo_mode: primero satisface cupos por sexo lo más equitativo posible.
    - Luego asigna el resto alternando para minimizar la diferencia de puntaje total.
    """
    df = poblacion_df.copy().reset_index(drop=True)
    # Map sexo a 'M'/'F' y comodin para NaN
    if sexo_mode and "SEXO" in df.columns:
        df["SEXO"] = normalizar_sexo(df["SEXO"])
        df["SEXO"] = df["SEXO"].fillna("COMODIN")
    else:
        df["SEXO"] = "NO_REQ"

    # Puntaje: si existe PUNTAJE usa, si no calcula promedio de skills si hay
    if "PUNTAJE" in df.columns:
        df["__PUNTAJE__"] = pd.to_numeric(df["PUNTAJE"], errors="coerce").fillna(0)
    else:
        # usamos promedio de columnas numéricas (habilidades)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["__PUNTAJE__"] = df[num_cols].mean(axis=1).fillna(0)

    # Inicializo equipos
    team_a = []
    team_b = []

    # Remover cabezas del pool y colocarlas
    pool = df.copy()
    if cabeza_a_name:
        row = pool[pool["NOMBRE"] == cabeza_a_name]
        if not row.empty:
            team_a.append(row.index[0])
            pool = pool.drop(index=row.index[0])
    if cabeza_b_name:
        row = pool[pool["NOMBRE"] == cabeza_b_name]
        if not row.empty:
            team_b.append(row.index[0])
            pool = pool.drop(index=row.index[0])

    # Si modo mixto: repartir por sexo primero
    if sexo_mode:
        males = pool[pool["SEXO"] == "M"].copy()
        fems = pool[pool["SEXO"] == "F"].copy()
        comodines = pool[pool["SEXO"] == "COMODIN"].copy()

        # Deseados por sexo en cada equipo (intento de igualdad)
        total_m = len(males)
        total_f = len(fems)
        desired_m_a = total_m // 2
        desired_m_b = total_m - desired_m_a
        desired_f_a = total_f // 2
        desired_f_b = total_f - desired_f_a

        # Ajustar si cabezas ya cuentan como M/F
        # contamos cuántos M/F ya en team_a/team_b (si se asignaron cabezas)
        def count_sex_in(team_indices, sex):
            if not team_indices: return 0
            return int(df.loc[team_indices]["SEXO"].eq(sex).sum())

        # Si cabezas (asignadas por nombre) estaban en pool y tenían sexo, descontar de desired
        # (pero las cabezas ya fueron quitadas del pool y añadidas a equipos)
        # Recalculo quotas restando los que ya estén en cada equipo
        m_in_a = count_sex_in(team_a, "M")
        m_in_b = count_sex_in(team_b, "M")
        f_in_a = count_sex_in(team_a, "F")
        f_in_b = count_sex_in(team_b, "F")
        desired_m_a = max(0, desired_m_a - m_in_a)
        desired_m_b = max(0, desired_m_b - m_in_b)
        desired_f_a = max(0, desired_f_a - f_in_a)
        desired_f_b = max(0, desired_f_b - f_in_b)

        # Función para asignar desde un pool ordenado por puntaje intentando balance de puntaje
        def asignar_desde(subpool_df, desired_a, desired_b):
            nonlocal team_a, team_b
            # sort descending puntaje to distribute stronger ones alternately / balancing by total puntaje
            sub = subpool_df.sort_values("__PUNTAJE__", ascending=False)
            for idx in sub.index:
                if desired_a > 0 or desired_b > 0:
                    # calcular puntajes actuales
                    punt_a = df.loc[team_a]["__PUNTAJE__"].sum() if team_a else 0
                    punt_b = df.loc[team_b]["__PUNTAJE__"].sum() if team_b else 0
                    # Prioriza ubicación donde menor puntaje actual y donde quota no esté completa
                    if (punt_a <= punt_b and desired_a > 0) or desired_b == 0:
                        team_a.append(idx); desired_a -= 1
                    else:
                        team_b.append(idx); desired_b -= 1
                else:
                    # quotas cumplidas; dejar para asignación general
                    continue
            return desired_a, desired_b

        desired_m_a, desired_m_b = asignar_desde(males, desired_m_a, desired_m_b)
        desired_f_a, desired_f_b = asignar_desde(fems, desired_f_a, desired_f_b)

        # Actualizo pool: quitar los ya asignados
        assigned_indices = set(team_a + team_b)
        remaining = pool.drop(index=list(assigned_indices.intersection(pool.index)))  # indices que siguen en pool
        # Añadir comodines y remaining para asignación final
        remaining = pd.concat([remaining, comodines]).drop_duplicates()

    else:
        # No sexo mode: remaining es pool después de extraer cabezas
        assigned_indices = set(team_a + team_b)
        remaining = pool.drop(index=list(assigned_indices.intersection(pool.index)))

    # Ahora necesitamos completar hasta size_a y size_b intentando balancear puntaje
    size_a_req = size_a
    size_b_req = size_b

    # Ya puestos algunos indices en teams, convertimos a sets
    team_a_set = set(team_a)
    team_b_set = set(team_b)

    # function to compute current puntaje sum
    def sum_puntaje(indices):
        if not indices: return 0
        return df.loc[list(indices)]["__PUNTAJE__"].sum()

    # ordeno remaining por puntaje desc para asignar mejores equitativamente
    remaining = remaining.sort_values("__PUNTAJE__", ascending=False)

    for idx in remaining.index:
        if len(team_a_set) < size_a_req and len(team_b_set) < size_b_req:
            # ambos equipos tienen espacio: asigno al que menor suma de puntaje tenga
            if sum_puntaje(team_a_set) <= sum_puntaje(team_b_set):
                team_a_set.add(idx)
            else:
                team_b_set.add(idx)
        elif len(team_a_set) < size_a_req:
            team_a_set.add(idx)
        elif len(team_b_set) < size_b_req:
            team_b_set.add(idx)
        else:
            # ambos llenos (ocurre si hay exceso) -> no asigno
            break

    # Finalmente, construyo dataframes
    equipo_a_df = df.loc[sorted(list(team_a_set))].copy().reset_index(drop=True)
    equipo_b_df = df.loc[sorted(list(team_b_set))].copy().reset_index(drop=True)

    # limpiar columna auxiliar
    equipo_a_df = equipo_a_df.drop(columns=["__PUNTAJE__"], errors="ignore")
    equipo_b_df = equipo_b_df.drop(columns=["__PUNTAJE__"], errors="ignore")

    return equipo_a_df, equipo_b_df

def generar_equipos(df, cabeza_a=None, cabeza_b=None, tipo_futbol="Mixto"):
    """
    Nueva lógica que respeta prioridades:
    1) Colocar cabezas (verificación de sexo si ambos seleccionados: deben ser mismo sexo)
    2) Garantizar cantidad igual o ±1
    3) Si Mixto: asegurar reparto equitativo por sexo
    4) Balancear puntaje agregando resto
    """
    df_work = df.copy().reset_index(drop=True)

    # Normalizar columna SEXO si existe
    if "SEXO" in df_work.columns:
        df_work["SEXO"] = normalizar_sexo(df_work["SEXO"])

    n = len(df_work)
    if n == 0:
        return df_work, df_work

    # Requerimientos de tamaño
    size_a = n // 2
    size_b = n - size_a

    # Validación cabezas: si ambos seleccionados y ambos existen y tipo Mixto -> deben ser mismo sexo
    if cabeza_a and cabeza_b:
        ca = df_work.loc[df_work["NOMBRE"] == cabeza_a]
        cb = df_work.loc[df_work["NOMBRE"] == cabeza_b]
        if not ca.empty and not cb.empty:
            sex_a = ca["SEXO"].iloc[0] if "SEXO" in ca.columns else None
            sex_b = cb["SEXO"].iloc[0] if "SEXO" in cb.columns else None
            # si tipo Mixto requerimos que sean mismos sexo
            if tipo_futbol == "Mixto" and pd.notna(sex_a) and pd.notna(sex_b) and sex_a != sex_b:
                st.error("Las cabezas de serie deben ser del mismo sexo en modo Mixto. Por favor seleccioná dos cabezas del mismo sexo o cambiá el tipo de fútbol.")
                return pd.DataFrame(), pd.DataFrame()

    # Si no es mixto: ignorar columna SEXO por completo
    sexo_mode = True if tipo_futbol == "Mixto" else False

    # Llamo a la asignación greedy que hace el trabajo pesado
    equipo_a, equipo_b = _asignacion_greedy_balance(df_work, size_a, size_b, cabeza_a, cabeza_b, sexo_mode=sexo_mode)

    # Como paso final, si no se cumplió exactamente tamaño por X motivo, ajusto para asegurar ±1
    # Rebalance simple: mover jugadores si hay diferencia >1
    def ajustar_tamanos(a_df, b_df):
        a = a_df.copy(); b = b_df.copy()
        while abs(len(a) - len(b)) > 1:
            if len(a) > len(b):
                # mover el jugador con menor puntaje de a a b
                if "PUNTAJE" in a.columns:
                    idx = a["PUNTAJE"].astype(float).idxmin()
                else:
                    # usa primera fila
                    idx = a.index[0]
                mover = a.loc[[idx]]
                b = pd.concat([b, mover], ignore_index=True)
                a = a.drop(index=[idx]).reset_index(drop=True)
            else:
                if "PUNTAJE" in b.columns:
                    idx = b["PUNTAJE"].astype(float).idxmin()
                else:
                    idx = b.index[0]
                mover = b.loc[[idx]]
                a = pd.concat([a, mover], ignore_index=True)
                b = b.drop(index=[idx]).reset_index(drop=True)
        return a.reset_index(drop=True), b.reset_index(drop=True)

    equipo_a, equipo_b = ajustar_tamanos(equipo_a, equipo_b)

    return equipo_a, equipo_b

# ------------------------------
# CARGA DE DATOS
# ------------------------------
archivo = st.file_uploader("📂 Cargar archivo .xlsx", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)
else:
    # tabla editable vacía
    default_df = pd.DataFrame(columns=columnas_esperadas)
    column_config = {
        "POSICION": st.column_config.SelectboxColumn("Posición", options=["ATAQUE", "DEFENSA", "ARQUERO"], required=True),
    }
    if "SEXO" in columnas_esperadas:
        column_config["SEXO"] = st.column_config.SelectboxColumn("Sexo", options=["M", "F"], required=True)
    if "PUNTAJE" in columnas_esperadas:
        column_config["PUNTAJE"] = st.column_config.NumberColumn(
            "Puntaje",
            help="De 0 a 100. Rango: 90-100 = 'Crack', 70-89 = 'Muy bueno', 50-69 = 'Promedio', <50 = 'A mejorar'",
            min_value=0, max_value=100, step=1
        )

    df = st.data_editor(
        default_df,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
    )

# Si tipo no mixto, ocultamos/ignoramos SEXO: si vino con archivo, eliminamos columna SEXO en memoria para evitar confusiones
if tipo_futbol != "Mixto" and "SEXO" in df.columns:
    # eliminamos columna SEXO de la vista de procesamiento (pero podremos mantener en el archivo original si querés)
    df = df.drop(columns=["SEXO"])

# ------------------------------
# MOSTRAR LA TABLA CON FORMATO CONDICIONAL (PUNTAJE)
# ------------------------------
if not df.empty:
    st.markdown("#### Tabla cargada (vista previa)")

    # Aplicar formato condicional sobre PUNTAJE si existe
    if "PUNTAJE" in df.columns:
        def color_puntaje(val):
            try:
                v = float(val)
            except:
                return ""
            if v >= 90: return 'background-color: #0b6623; color: white'   # verde oscuro
            if v >= 70: return 'background-color: #66bb6a; color: black'   # verde medio
            if v >= 50: return 'background-color: #ffcc66; color: black'   # naranja claro
            return 'background-color: #ff6666; color: white'               # rojo

        styled = df.style.applymap(lambda v: color_puntaje(v) if isinstance(v, (int,float)) or (isinstance(v,str) and v.replace('.','',1).isdigit()) else "", subset=["PUNTAJE"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# ------------------------------
# GENERAR EQUIPOS
# ------------------------------
if all(col in df.columns for col in [c for c in columnas_esperadas if c in df.columns]) and not df.empty:
    st.success("✅ Datos listos para generar equipos")

    col1, col2 = st.columns(2)
    with col1:
        nombres = df["NOMBRE"].dropna().astype(str).tolist()
        cabeza_a = st.selectbox("Cabeza de grupo A (opcional)", options=[None] + nombres)
    with col2:
        cabeza_b = st.selectbox("Cabeza de grupo B (opcional)", options=[None] + nombres)

    # Si se eligen dos cabezas del mismo nombre prevenir duplicado
    if cabeza_a is not None and cabeza_b is not None and cabeza_a == cabeza_b:
        st.warning("No podés elegir la misma cabeza para A y B. Elegí otro jugador.")
        cabeza_b = None

    if st.button("🎲 Generar Equipos", type="primary"):
        with st.spinner("Generando equipos balanceados..."):
            equipo_a, equipo_b = generar_equipos(df, cabeza_a, cabeza_b, tipo_futbol)

        if equipo_a.empty and equipo_b.empty:
            st.error("No se generaron equipos (ver mensaje de error arriba).")
        else:
            # Mostrar equipos
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🟥 Equipo A")
                for _, row in equipo_a.iterrows():
                    pos = row.get("POSICION", "")
                    st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>📌 {pos}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("### 🟦 Equipo B")
                for _, row in equipo_b.iterrows():
                    pos = row.get("POSICION", "")
                    st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>📌 {pos}</div>", unsafe_allow_html=True)

            # Estadísticas y gráficos
            stats_a = resumir_stats(equipo_a)
            stats_b = resumir_stats(equipo_b)

            # Gráfico de medias (para Express usa PUNTAJE, para Avanzado usa las skills presentes)
            if not stats_a.empty and not stats_b.empty:
                radar_df = pd.DataFrame({
                    "Variable": list(stats_a.index) + list(stats_b.index),
                    "Media": list(stats_a.values) + list(stats_b.values),
                    "Equipo": ["A"]*len(stats_a) + ["B"]*len(stats_b)
                })
                fig_media = px.bar(radar_df, x="Variable", y="Media", color="Equipo", barmode="group", title="Promedio de variables por equipo")
                st.plotly_chart(fig_media, use_container_width=True)

            # Promedio de PUNTAJE (si existe) para mostrar comparativa simple
            if "PUNTAJE" in equipo_a.columns or "PUNTAJE" in equipo_b.columns:
                pa = equipo_a["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_a.columns else np.nan
                pb = equipo_b["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_b.columns else np.nan
                df_prom = pd.DataFrame({
                    "Equipo":["A","B"],
                    "Promedio PUNTAJE":[pa, pb]
                })
                fig_prom = px.bar(df_prom, x="Equipo", y="Promedio PUNTAJE", title="Promedio de PUNTAJE por equipo", text="Promedio PUNTAJE")
                st.plotly_chart(fig_prom, use_container_width=True)

            # Distribución de sexo por equipo (solo si Mixto o si existe columna SEXO)
            if tipo_futbol == "Mixto" or "SEXO" in pd.concat([equipo_a, equipo_b]).columns:
                sex_a = equipo_a.get("SEXO", pd.Series(dtype=str)).fillna("COMODIN")
                sex_b = equipo_b.get("SEXO", pd.Series(dtype=str)).fillna("COMODIN")
                df_sex = pd.concat([pd.DataFrame({"SEXO": sex_a, "Equipo":"A"}), pd.DataFrame({"SEXO": sex_b, "Equipo":"B"})], ignore_index=True)
                # normalizar display
                df_sex["SEXO"] = df_sex["SEXO"].fillna("COMODIN")
                fig_sex = px.histogram(df_sex, x="SEXO", color="Equipo", barmode="group", title="Distribución de sexo por equipo")
                st.plotly_chart(fig_sex, use_container_width=True)

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

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
        .footer {
            text-align: center;
            margin-top: 1.5rem;
            margin-bottom: 2rem;
            color: #333;
            font-size: 1.15rem; /* aumentado */
        }
        .small-muted { color: #666; font-size:0.9rem; }
        .center { text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Generador de Equipos Balanceados")
st.caption("**Build your Team APP**")
st.markdown("Usá el wizard: seleccioná modo → cargá datos → (opcional) configurá cabezas → generá equipos y descargá.")

# ------------------------------
# Helpers + estado
# ------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "df_original" not in st.session_state:
    st.session_state.df_original = pd.DataFrame()
if "df_work" not in st.session_state:
    st.session_state.df_work = pd.DataFrame()
if "equipo_a" not in st.session_state:
    st.session_state.equipo_a = pd.DataFrame()
if "equipo_b" not in st.session_state:
    st.session_state.equipo_b = pd.DataFrame()

def go_next():
    st.session_state.step += 1

def go_back():
    st.session_state.step = max(1, st.session_state.step - 1)

# ------------------------------
# Configuración inicial: Modo y Tipo
# ------------------------------
with st.expander("Paso 1 — Configuración", expanded=True):
    modo = st.radio("Seleccioná el modo:", ["Express", "Avanzado"], horizontal=True, key="modo_radio")
    tipo_futbol = st.selectbox("Tipo de fútbol:", ["Mixto", "Masculino", "Femenino"], key="tipo_select")
    st.markdown('<div class="modo-banner {}">{}</div>'.format(
        "modo-express" if modo == "Express" else "modo-avanzado",
        "🟢 Estás en MODO EXPRESS" if modo == "Express" else "🔵 Estás en MODO AVANZADO"
    ), unsafe_allow_html=True)
    st.markdown("Siguiente paso: cargar datos de jugadores.")
    st.button("Siguiente → Paso 2 (Cargar datos)", on_click=lambda: setattr(st.session_state, "step", 2))

# ------------------------------
# Columnas esperadas según modo/tipo
# ------------------------------
def columnas_esperadas_for(modo, tipo):
    if modo == "Express":
        cols = ["NOMBRE", "POSICION", "PUNTAJE"]
    else:
        cols = ["NOMBRE", "POSICION", "EDAD", "RESISTENCIA", "VELOCIDAD", "DRIBLE", "PEGADA", "PASE", "DEFENSA"]
    if tipo == "Mixto":
        cols = ["NOMBRE", "SEXO"] + [c for c in cols if c != "NOMBRE"]
    return cols

# ------------------------------
# UTILIDAD: normalizar sexo
# ------------------------------
def normalizar_sexo(col):
    m = {"M": "M", "H": "M", "HOMBRE": "M", "MAS": "M", "F": "F", "MUJER": "F", "W": "F"}
    return col.astype(str).str.upper().map(m).where(lambda x: x.isin(["M","F"]), other=np.nan)

# ------------------------------
# Funciones de asignación (la misma lógica que validada)
# ------------------------------
def _asignacion_greedy_balance(poblacion_df, size_a, size_b, cabeza_a_name=None, cabeza_b_name=None, sexo_mode=False):
    df = poblacion_df.copy().reset_index(drop=True)
    if sexo_mode and "SEXO" in df.columns:
        df["SEXO"] = normalizar_sexo(df["SEXO"])
        df["SEXO"] = df["SEXO"].fillna("COMODIN")
    else:
        df["SEXO"] = "NO_REQ"

    if "PUNTAJE" in df.columns:
        df["__PUNTAJE__"] = pd.to_numeric(df["PUNTAJE"], errors="coerce").fillna(0)
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["__PUNTAJE__"] = df[num_cols].mean(axis=1).fillna(0)

    team_a = []
    team_b = []
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

    if sexo_mode:
        males = pool[pool["SEXO"] == "M"].copy()
        fems = pool[pool["SEXO"] == "F"].copy()
        comodines = pool[pool["SEXO"] == "COMODIN"].copy()

        total_m = len(males)
        total_f = len(fems)
        desired_m_a = total_m // 2
        desired_m_b = total_m - desired_m_a
        desired_f_a = total_f // 2
        desired_f_b = total_f - desired_f_a

        def count_sex_in(team_indices, sex):
            if not team_indices: return 0
            return int(df.loc[team_indices]["SEXO"].eq(sex).sum())

        m_in_a = count_sex_in(team_a, "M")
        m_in_b = count_sex_in(team_b, "M")
        f_in_a = count_sex_in(team_a, "F")
        f_in_b = count_sex_in(team_b, "F")
        desired_m_a = max(0, desired_m_a - m_in_a)
        desired_m_b = max(0, desired_m_b - m_in_b)
        desired_f_a = max(0, desired_f_a - f_in_a)
        desired_f_b = max(0, desired_f_b - f_in_b)

        def asignar_desde(subpool_df, desired_a, desired_b):
            nonlocal team_a, team_b
            sub = subpool_df.sort_values("__PUNTAJE__", ascending=False)
            for idx in sub.index:
                if desired_a > 0 or desired_b > 0:
                    punt_a = df.loc[team_a]["__PUNTAJE__"].sum() if team_a else 0
                    punt_b = df.loc[team_b]["__PUNTAJE__"].sum() if team_b else 0
                    if (punt_a <= punt_b and desired_a > 0) or desired_b == 0:
                        team_a.append(idx); desired_a -= 1
                    else:
                        team_b.append(idx); desired_b -= 1
                else:
                    continue
            return desired_a, desired_b

        desired_m_a, desired_m_b = asignar_desde(males, desired_m_a, desired_m_b)
        desired_f_a, desired_f_b = asignar_desde(fems, desired_f_a, desired_f_b)

        assigned_indices = set(team_a + team_b)
        remaining = pool.drop(index=list(assigned_indices.intersection(pool.index)))
        remaining = pd.concat([remaining, comodines]).drop_duplicates()
    else:
        assigned_indices = set(team_a + team_b)
        remaining = pool.drop(index=list(assigned_indices.intersection(pool.index)))

    size_a_req = size_a
    size_b_req = size_b
    team_a_set = set(team_a)
    team_b_set = set(team_b)

    def sum_puntaje(indices):
        if not indices: return 0
        return df.loc[list(indices)]["__PUNTAJE__"].sum()

    remaining = remaining.sort_values("__PUNTAJE__", ascending=False)

    for idx in remaining.index:
        if len(team_a_set) < size_a_req and len(team_b_set) < size_b_req:
            if sum_puntaje(team_a_set) <= sum_puntaje(team_b_set):
                team_a_set.add(idx)
            else:
                team_b_set.add(idx)
        elif len(team_a_set) < size_a_req:
            team_a_set.add(idx)
        elif len(team_b_set) < size_b_req:
            team_b_set.add(idx)
        else:
            break

    equipo_a_df = df.loc[sorted(list(team_a_set))].copy().reset_index(drop=True)
    equipo_b_df = df.loc[sorted(list(team_b_set))].copy().reset_index(drop=True)
    equipo_a_df = equipo_a_df.drop(columns=["__PUNTAJE__"], errors="ignore")
    equipo_b_df = equipo_b_df.drop(columns=["__PUNTAJE__"], errors="ignore")
    return equipo_a_df, equipo_b_df

def generar_equipos(df, cabeza_a=None, cabeza_b=None, tipo_futbol="Mixto"):
    df_work = df.copy().reset_index(drop=True)
    if "SEXO" in df_work.columns:
        df_work["SEXO"] = normalizar_sexo(df_work["SEXO"])
    n = len(df_work)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()
    size_a = n // 2
    size_b = n - size_a

    if cabeza_a and cabeza_b:
        ca = df_work.loc[df_work["NOMBRE"] == cabeza_a]
        cb = df_work.loc[df_work["NOMBRE"] == cabeza_b]
        if not ca.empty and not cb.empty:
            sex_a = ca["SEXO"].iloc[0] if "SEXO" in ca.columns else None
            sex_b = cb["SEXO"].iloc[0] if "SEXO" in cb.columns else None
            if tipo_futbol == "Mixto" and pd.notna(sex_a) and pd.notna(sex_b) and sex_a != sex_b:
                st.error("Las cabezas de serie deben ser del mismo sexo en modo Mixto. Por favor seleccioná dos cabezas del mismo sexo o cambiá el tipo de fútbol.")
                return pd.DataFrame(), pd.DataFrame()

    sexo_mode = True if tipo_futbol == "Mixto" else False
    equipo_a, equipo_b = _asignacion_greedy_balance(df_work, size_a, size_b, cabeza_a, cabeza_b, sexo_mode=sexo_mode)

    def ajustar_tamanos(a_df, b_df):
        a = a_df.copy(); b = b_df.copy()
        while abs(len(a) - len(b)) > 1:
            if len(a) > len(b):
                if "PUNTAJE" in a.columns:
                    idx = a["PUNTAJE"].astype(float).idxmin()
                else:
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
# Paso 2 — Carga de datos (y edición)
# ------------------------------
if st.session_state.step == 2:
    st.header("Paso 2 — Cargar o editar jugadores")
    st.markdown("Subí un archivo .xlsx o cargá los jugadores en la tabla editable. (La columna SEXO aparece solo si elegiste 'Mixto' en el paso 1.)")
    columnas_esperadas = columnas_esperadas_for(modo, tipo_futbol)

    archivo = st.file_uploader("📂 Cargar archivo .xlsx (opcional)", type=["xlsx"], key="uploader_step2")
    if archivo:
        try:
            df = pd.read_excel(archivo)
            # aseguramos columnas mínimas
            st.session_state.df_original = df.copy()
            # Si no viene SEXO pero tipo Mixto, dejamos como NaN (usuario debe completar)
            # Mantener df_work para edición y procesamiento
            st.session_state.df_work = df.copy()
            st.success("Archivo cargado. Podés editar abajo si necesitás.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.session_state.df_work = pd.DataFrame(columns=columnas_esperadas)
    else:
        # tabla editable vacía
        default_df = pd.DataFrame(columns=columnas_esperadas)
        column_config = {
            "POSICION": st.column_config.SelectboxColumn("Posición", options=["ATAQUE", "MEDIO", "DEFENSA", "ARQUERO"], required=True),
        }
        if "SEXO" in columnas_esperadas:
            column_config["SEXO"] = st.column_config.SelectboxColumn("Sexo", options=["M", "F"], required=True)
        if "PUNTAJE" in columnas_esperadas:
            column_config["PUNTAJE"] = st.column_config.NumberColumn(
                "Puntaje", help="0-100", min_value=0, max_value=100, step=1
            )

        edited = st.data_editor(default_df, column_config=column_config, num_rows="dynamic", use_container_width=True)
        st.session_state.df_work = edited.copy()
        st.session_state.df_original = edited.copy()

    # Vista con formato condicional (puntaje) después de la carga
    if not st.session_state.df_work.empty:
        st.markdown("#### Vista previa (formato condicional en PUNTAJE)")
        df_preview = st.session_state.df_work.copy()
        if "PUNTAJE" in df_preview.columns:
            def color_puntaje(val):
                try:
                    v = float(val)
                except:
                    return ""
                if v >= 90: return 'background-color: #0b6623; color: white'
                if v >= 70: return 'background-color: #66bb6a; color: black'
                if v >= 50: return 'background-color: #ffcc66; color: black'
                return 'background-color: #ff6666; color: white'
            styled = df_preview.style.applymap(lambda v: color_puntaje(v) if isinstance(v, (int,float)) or (isinstance(v,str) and v.replace('.','',1).isdigit()) else "", subset=["PUNTAJE"])
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(df_preview, use_container_width=True)

    cols = st.columns([1,1,1])
    with cols[0]:
        st.button("← Volver (Paso 1)", on_click=lambda: setattr(st.session_state, "step", 1))
    with cols[1]:
        # Botón de generar también disponible desde paso 2 (opción B)
        st.button("🎲 Generar equipos (puedes configurar cabezas en Paso 3)", on_click=lambda: setattr(st.session_state, "step", 4))
    with cols[2]:
        st.button("Siguiente → Paso 3 (Configurar cabezas)", on_click=lambda: setattr(st.session_state, "step", 3))

# ------------------------------
# Paso 3 — Cabezas de serie (opcional)
# ------------------------------
if st.session_state.step == 3:
    st.header("Paso 3 — (Opcional) Elegí cabezas de serie")
    df_work = st.session_state.df_work.copy()
    if df_work.empty:
        st.warning("Primero cargá o edita la tabla en el Paso 2.")
        if st.button("← Volver (Paso 2)"): st.session_state.step = 2
    else:
        nombres = df_work["NOMBRE"].dropna().astype(str).tolist()
        col1, col2, col3 = st.columns(3)
        with col1:
            cabeza_a = st.selectbox("Cabeza de grupo A (opcional)", options=[None] + nombres, key="cabeza_a")
        with col2:
            cabeza_b = st.selectbox("Cabeza de grupo B (opcional)", options=[None] + nombres, key="cabeza_b")
        with col3:
            st.markdown("Si no elegís cabezas, el algoritmo distribuirá libremente.")
        st.markdown("Recordá: en modo Mixto las cabezas deben ser del mismo sexo.")
        cols = st.columns([1,1,1])
        with cols[0]:
            st.button("← Volver (Paso 2)", on_click=lambda: setattr(st.session_state, "step", 2))
        with cols[1]:
            st.button("Generar equipos ahora", on_click=lambda: setattr(st.session_state, "step", 4))
        with cols[2]:
            st.button("Saltar y Generar →", on_click=lambda: setattr(st.session_state, "step", 4))

# ------------------------------
# Paso 4 — Resultados y descargas
# ------------------------------
if st.session_state.step == 4:
    st.header("Paso 4 — Resultados")
    df_work = st.session_state.df_work.copy()
    if df_work.empty:
        st.error("No hay datos cargados. Volvé al Paso 2 y cargá jugadores.")
        if st.button("← Volver (Paso 2)"): st.session_state.step = 2
    else:
        # Recupero cabezas si existieran
        cabeza_a = st.session_state.get("cabeza_a", None)
        cabeza_b = st.session_state.get("cabeza_b", None)

        # Si tipo no mixto, eliminamos SEXO para el cálculo (pero mantenemos df_original intacto)
        df_for_algo = df_work.copy()
        if tipo_futbol != "Mixto" and "SEXO" in df_for_algo.columns:
            df_for_algo = df_for_algo.drop(columns=["SEXO"])

        # Validaciones antes de generar
        valid = True
        if tipo_futbol == "Mixto":
            # comprobar que exista columna SEXO y no haya NaN
            if "SEXO" not in df_work.columns:
                st.error("En modo Mixto necesitás la columna SEXO completa. Volvé al Paso 2 y completala.")
                valid = False
            else:
                s = normalizar_sexo(df_work["SEXO"])
                if s.isna().any():
                    st.error("Hay jugadores sin SEXO definido. Completá la columna SEXO para proceder en modo Mixto.")
                    valid = False

        if not valid:
            if st.button("← Volver (Paso 2)"): st.session_state.step = 2
        else:
            # Genero equipos
            equipo_a, equipo_b = generar_equipos(df_for_algo, cabeza_a, cabeza_b, tipo_futbol)
            st.session_state.equipo_a = equipo_a.copy()
            st.session_state.equipo_b = equipo_b.copy()

            if equipo_a.empty and equipo_b.empty:
                st.error("No se generaron equipos (posible conflicto con cabezas). Volvé a revisar las selecciones.")
                if st.button("← Volver (Paso 3)"): st.session_state.step = 3
            else:
                # Tarjetas resumen arriba
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.markdown(f"**Jugadores totales:** {len(df_for_algo)}")
                with col_s2:
                    st.markdown(f"**Equipo A (jugadores):** {len(equipo_a)}")
                    st.markdown(f"**Equipo B (jugadores):** {len(equipo_b)}")
                with col_s3:
                    # Promedios PUNTAJE
                    pa = equipo_a["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_a.columns else np.nan
                    pb = equipo_b["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_b.columns else np.nan
                    diff = None
                    try:
                        diff = float(round(abs(pa - pb), 2))
                    except:
                        diff = None
                    st.markdown(f"**Diferencia de promedio:** {diff if diff is not None else 'N/A'}")

                # Mostrar listas
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

                # Gráfico: Promedio general de PUNTAJE (una barra por equipo)
                if "PUNTAJE" in equipo_a.columns or "PUNTAJE" in equipo_b.columns:
                    pa = equipo_a["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_a.columns else np.nan
                    pb = equipo_b["PUNTAJE"].astype(float).mean() if "PUNTAJE" in equipo_b.columns else np.nan
                    df_prom = pd.DataFrame({"Equipo": ["A", "B"], "Promedio": [pa, pb]})
                    fig_prom = px.bar(df_prom, x="Equipo", y="Promedio", color="Equipo",
                                      color_discrete_map={"A":"#0b63d6","B":"#d62828"},
                                      title="Promedio general de PUNTAJE por equipo",
                                      text="Promedio")
                    fig_prom.update_traces(texttemplate='%{text:.2f}', textposition='outside', textfont_size=20)
                    fig_prom.update_layout(yaxis_range=[0,100], title_font_size=18, uniformtext_minsize=12)
                    st.plotly_chart(fig_prom, use_container_width=True)

                # Distribución sexo (si aplica)
                if tipo_futbol == "Mixto" or "SEXO" in pd.concat([equipo_a, equipo_b]).columns:
                    sex_a = equipo_a.get("SEXO", pd.Series(dtype=str)).fillna("COMODIN")
                    sex_b = equipo_b.get("SEXO", pd.Series(dtype=str)).fillna("COMODIN")
                    df_sex = pd.concat([
                        pd.DataFrame({"SEXO": sex_a, "Equipo":"A"}),
                        pd.DataFrame({"SEXO": sex_b, "Equipo":"B"})
                    ], ignore_index=True)
                    df_sex["SEXO"] = df_sex["SEXO"].fillna("COMODIN")
                    fig_sex = px.histogram(df_sex, x="SEXO", color="Equipo", barmode="group", title="Distribución de sexo por equipo")
                    st.plotly_chart(fig_sex, use_container_width=True)

                # Descargas: equipos y listado original (todas las columnas)
                # 1) Export equipos
                output = io.BytesIO()
                try:
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        equipo_a.to_excel(writer, sheet_name="Equipo A", index=False)
                        equipo_b.to_excel(writer, sheet_name="Equipo B", index=False)
                    data_bytes = output.getvalue()
                except Exception:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        equipo_a.to_excel(writer, sheet_name="Equipo A", index=False)
                        equipo_b.to_excel(writer, sheet_name="Equipo B", index=False)
                    data_bytes = output.getvalue()

                st.download_button(
                    label="📥 Descargar equipos en Excel",
                    data=data_bytes,
                    file_name="equipos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # 2) Descargar listado original cargado (todas las columnas) - aparece SOLO al final
                original_df = st.session_state.df_original.copy()
                # Si original está vacío (porque el usuario editó en data_editor), guardamos df_work como original
                if original_df.empty:
                    original_df = st.session_state.df_work.copy()

                out2 = io.BytesIO()
                try:
                    with pd.ExcelWriter(out2, engine="xlsxwriter") as writer:
                        original_df.to_excel(writer, sheet_name="Jugadores", index=False)
                    bytes2 = out2.getvalue()
                except Exception:
                    out2 = io.BytesIO()
                    with pd.ExcelWriter(out2, engine="openpyxl") as writer:
                        original_df.to_excel(writer, sheet_name="Jugadores", index=False)
                    bytes2 = out2.getvalue()

                st.download_button(
                    label="📥 Descargar jugadores cargados para reutilizar",
                    data=bytes2,
                    file_name="jugadores_cargados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Botones de navegación final
                cols_nav = st.columns([1,1,1])
                with cols_nav[0]:
                    st.button("← Volver (Paso 3)", on_click=lambda: setattr(st.session_state, "step", 3))
                with cols_nav[1]:
                    st.button("Nueva sesión (iniciar)", on_click=lambda: [setattr(st.session_state, "step", 1),
                                                                           st.session_state.df_original.clear(), st.session_state.df_work.clear(),
                                                                           st.session_state.equipo_a.clear() if isinstance(st.session_state.equipo_a, pd.DataFrame) else None,
                                                                           st.session_state.equipo_b.clear() if isinstance(st.session_state.equipo_b, pd.DataFrame) else None])
                with cols_nav[2]:
                    st.button("Copiar resultados a clipboard (tabla)", on_click=lambda: st.experimental_set_query_params())

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        ☕ <strong>Si te gustó esta app y querés invitarme un cafecito:</strong><br>
        <a href="https://cafecito.app/jmperalta" target="_blank">Cafecito</a>  
        <div class="small-muted">Contacto: <a href="https://www.linkedin.com/in/peraltajm/" target="_blank">LinkedIn</a></div>
    </div>
    """,
    unsafe_allow_html=True,
)

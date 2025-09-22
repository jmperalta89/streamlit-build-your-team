import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io
import base64
import textwrap
import streamlit.components.v1 as components

# Config
st.set_page_config(page_title="Build Your Team APP — Generador de Equipos", page_icon="⚽", layout="wide")

# --- Styles (tipografías y visual) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stButton>button { font-family: 'Inter', 'Poppins', sans-serif; }
.header { display:flex; align-items:center; gap:16px; }
.brand { font-weight:700; font-size:24px; }
.subtitle { color: #6b7280; }
.card { background: #ffffff; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08); }
.team-card { border-left: 6px solid #2563eb; padding:12px; border-radius:8px; }
.donate { background:#f59e0b; color:white; padding:10px 16px; border-radius:10px; font-weight:600; }
.small { font-size:13px; color:#6b7280; }
</style>
""", unsafe_allow_html=True)

# Header / Branding
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header">\n  <div class="brand">Build Your Team APP</div>\n  <div class="subtitle">Generador de equipos balanceados (K-Means)</div>\n</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="text-align:right">\n  <a href="https://cafecito.app/jmperalta" target="_blank" class="donate">Invitame un cafecito ☕</a>\n</div>', unsafe_allow_html=True)

st.markdown('---')

# Left column: carga y onboarding
left, right = st.columns([1,2])

with left:
    st.header("Cargar datos")

    # Onboarding / modelo descargable
    with st.expander("📘 Modelo y onboarding (ejemplo de archivo)", expanded=True):
        st.markdown("""
        **Formato esperado (columnas obligatorias):**
        - `NOMBRE` (string)
        - `POSICION` (string)
        - `EDAD` (número)
        - `RESISTENCIA`, `VELOCIDAD`, `DRIBLE`, `PEGADA`, `PASE`, `DEFENSA` (números, ideal 0-100)
        - `SEXO` (H/M o Hombre/Mujer o F)

        **Pasos de carga:**
        1. Descargá el modelo de Excel de ejemplo.\n2. Completá los jugadores.\n3. Subí el archivo o pegá la tabla en el editor de abajo.\n
        > Si tenés preguntas, contactame en LinkedIn: [peraltajm].
        """, unsafe_allow_html=True)

        # Generar un template de ejemplo para descargar
        template = pd.DataFrame({
            'NOMBRE': ['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4', 'Jugador 5', 'Jugador 6'],
            'POSICION': ['Delantero','Mediocentro','Defensa','Delantero','Arquero','Defensa'],
            'EDAD': [25,27,30,22,28,24],
            'RESISTENCIA':[80,70,65,75,50,68],
            'VELOCIDAD':[85,72,60,88,30,62],
            'DRIBLE':[82,74,58,80,20,55],
            'PEGADA':[78,66,55,83,10,40],
            'PASE':[70,80,50,65,30,60],
            'DEFENSA':[40,55,80,30,90,78],
            'SEXO':['H','H','H','M','H','M']
        })
        towrite = io.BytesIO()
        template.to_excel(towrite, index=False, sheet_name='ejemplo')
        towrite.seek(0)
        st.download_button(label="📥 Descargar modelo (.xlsx)", data=towrite, file_name="modelo_ejemplo_team.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")

    # File uploader with feedback
    archivo = st.file_uploader("Cargar archivo .xlsx", type=["xlsx"], help="Subí un archivo con la estructura del modelo, o pegá/edita abajo.")

    st.markdown("**O pegá / editá la tabla aquí**")
    # data_editor for manual editing
    default_df = template.copy()
    edited = st.experimental_data_editor(default_df, num_rows="dynamic")

    st.markdown("---")
    st.markdown("**Opciones de generación**")
    n_clusters = st.selectbox("Número de equipos", options=[2,3,4], index=0)
    random_state = st.number_input("Seed (random_state)", min_value=0, value=42, step=1)
    cabeza_a = st.text_input("Cabeza de grupo A (opcional)")
    cabeza_b = st.text_input("Cabeza de grupo B (opcional)")

    # Donación destacada
    st.markdown('<div class="card small">Si te sirvió la app, considerá invitarme un cafecito — ¡gracias por el apoyo! <a href="https://cafecito.app/jmperalta" target="_blank">Invitar</a></div>', unsafe_allow_html=True)

with right:
    st.header("Previsualización y control")
    # Choose source
    source = st.radio("Fuente de datos", options=["Subir archivo","Usar editor (tabla)"], index=1)

    if source == "Subir archivo":
        if archivo:
            with st.spinner('Cargando archivo...'):
                try:
                    df = pd.read_excel(archivo)
                    st.success('Archivo cargado correctamente')
                except Exception as e:
                    st.error(f'Error leyendo el archivo: {e}')
                    df = None
        else:
            st.info('No hay archivo cargado. Podés usar la tabla de la derecha o descargar el modelo.')
            df = None
    else:
        df = edited.copy()

    # Validate columns
    columnas_esperadas = ['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']
    if df is not None:
        missing = [c for c in columnas_esperadas if c not in df.columns]
        if missing:
            st.error(f'Faltan columnas: {missing}')
            st.stop()
        else:
            st.success('Estructura de columnas OK')
            st.dataframe(df.head(10))

    st.markdown('---')

    # Generar equipos
    if st.button('Generar Equipos', type='primary'):
        if df is None:
            st.error('No hay datos para generar equipos.')
        else:
            # Feedback: progress bar and spinner
            progress = st.progress(0)
            with st.spinner('Generando equipos con K-Means...'):
                # Preprocesamiento
                progress.progress(10)
                features = ['EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']
                df_proc = df.copy()
                # Normalizar SEXO
                df_proc['SEXO'] = df_proc['SEXO'].astype(str).str.upper().map({'M':1,'H':1,'HOMBRE':1,'F':0,'MUJER':0})
                df_proc['SEXO'] = df_proc['SEXO'].fillna(0.5)

                # Fillna numeric
                for f in features:
                    df_proc[f] = pd.to_numeric(df_proc[f], errors='coerce').fillna(0)

                scaler = MinMaxScaler()
                X = scaler.fit_transform(df_proc[features])
                progress.progress(30)

                # KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
                labels = kmeans.fit_predict(X)
                df_proc['cluster'] = labels
                progress.progress(55)

                # Build teams: try to balance number of players per cluster
                teams = {}
                for c in range(n_clusters):
                    teams[c] = df_proc[df_proc['cluster']==c].reset_index(drop=True)

                # If user provided "cabezas", force assign
                if cabeza_a:
                    mask = df_proc['NOMBRE']==cabeza_a
                    if mask.any():
                        df_proc.loc[mask, 'cluster'] = 0
                if cabeza_b and n_clusters>1:
                    mask = df_proc['NOMBRE']==cabeza_b
                    if mask.any():
                        df_proc.loc[mask, 'cluster'] = 1

                # After forcing, rebuild teams
                teams = {c: df_proc[df_proc['cluster']==c].reset_index(drop=True) for c in range(n_clusters)}
                progress.progress(75)

                # Create a simple balancing step: if clusters sizes differ a lot, sort by puntaje y redistribute alternately across teams
                sizes = [len(teams[c]) for c in range(n_clusters)]
                if max(sizes) - min(sizes) > 1:
                    # compute a puntaje global and repartition
                    puntajes = X.sum(axis=1)
                    df_sorted = df_proc.copy()
                    df_sorted['puntaje'] = puntajes
                    df_sorted = df_sorted.sort_values('puntaje', ascending=False).reset_index(drop=True)
                    # alternate assign
                    asign = [i % n_clusters for i in range(len(df_sorted))]
                    df_sorted['assigned_team'] = asign
                    teams = {c: df_sorted[df_sorted['assigned_team']==c].reset_index(drop=True) for c in range(n_clusters)}

                progress.progress(95)

                st.success('Equipos generados')
                progress.progress(100)

            st.toast('Generación completada ✅')

            # Mostrar equipos en cards
            st.header('Resultado — Equipos')
            cols = st.columns(n_clusters)

            # For export: prepare excel with multiple sheets
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for c in range(n_clusters):
                    teams[c].to_excel(writer, index=False, sheet_name=f'Equipo_{c+1}')
            processed_data = output.getvalue()

            st.download_button('📥 Exportar resultados (.xlsx)', data=processed_data, file_name='equipos_generados.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # Individual cards and copy buttons using components
            for i in range(n_clusters):
                with cols[i]:
                    st.markdown(f"<div class='card team-card'><h4>Equipo {i+1} — {len(teams[i])} jugadores</h4>", unsafe_allow_html=True)
                    st.dataframe(teams[i][['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']])

                    # Prepare CSV and txt for copy
                    csv_text = teams[i][['NOMBRE','POSICION','EDAD']].to_csv(index=False)
                    txt = '\n'.join(teams[i]['NOMBRE'].astype(str).tolist())
                    # Show copy button via a small HTML component
                    copy_html = f"""
                    <div style='margin-top:8px;'>
                      <button onclick="navigator.clipboard.writeText(`{txt}`)">Copiar al portapapeles</button>
                      <a href='data:text/csv;charset=utf-8,{base64.b64encode(csv_text.encode()).decode()}' download='equipo_{i+1}.csv'>Descargar CSV</a>
                    </div>
                    """
                    components.html(copy_html, height=60)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Radar chart comparativo entre equipos 1 y 2 (si existen)
            try:
                def resumir_stats(equipo_df):
                    return equipo_df[['RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA']].mean()

                if n_clusters >= 2:
                    stats_list = [resumir_stats(teams[c]) for c in range(min(4,n_clusters))]
                    radar_df = pd.DataFrame({f'Equipo {i+1}': s.values for i,s in enumerate(stats_list)}, index=stats_list[0].index)
                    radar_long = radar_df.reset_index().melt(id_vars='index', var_name='Equipo', value_name='Valor')
                    radar_long = radar_long.rename(columns={'index':'Habilidad'})
                    fig = px.line_polar(radar_long, r='Valor', theta='Habilidad', color='Equipo', line_close=True)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f'No se pudo generar el radar chart: {e}')

            # Link de contacto
            st.markdown('---')
            st.markdown('¿Dudas o querés mejorar el algoritmo? Contactame en LinkedIn: [peraltajm](https://www.linkedin.com/in/peraltajm/)')

# Footer
st.markdown('---')
st.markdown('Creado por Juan Manuel — Build Your Team APP · [LinkedIn](https://www.linkedin.com/in/peraltajm/)')

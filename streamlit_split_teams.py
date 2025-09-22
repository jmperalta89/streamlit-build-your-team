import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io
import base64
import streamlit.components.v1 as components

# Config
st.set_page_config(page_title="Build Your Team APP — Generador de Equipos", page_icon="⚽", layout="wide")

# --- Styles ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stButton>button { font-family: 'Inter', 'Poppins', sans-serif; }
.header { display:flex; align-items:center; gap:16px; }
.brand { font-weight:700; font-size:24px; }
.subtitle { color: #6b7280; }
.card { background: #ffffff; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08); }
.team-card { border-left: 6px solid #2563eb; padding:12px; border-radius:8px; }
.donate { background:#6366f1; color:white; padding:10px 16px; border-radius:10px; font-weight:600; text-decoration:none; }
.small { font-size:13px; color:#6b7280; }
</style>
""", unsafe_allow_html=True)

# Header / Branding
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header">\n  <div class="brand">Build Your Team APP ⚽</div>\n  <div class="subtitle">Generador de equipos balanceados (K-Means)</div>\n</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="text-align:right">\n  <a href="https://cafecito.app/jmperalta" target="_blank" class="donate">☕ Invitame un cafecito</a>\n</div>', unsafe_allow_html=True)

st.markdown('---')

# Selector de modo
modo = st.radio("Seleccioná el modo de carga", ["Express","Avanzado"], index=0)

# --- Columnas y template según modo ---
if modo == "Express":
    columnas_esperadas = ['NOMBRE','SEXO','RESISTENCIA','HABILIDAD','POSICION']
    template = pd.DataFrame({
        'NOMBRE': ['Jugador 1','Jugador 2','Jugador 3','Jugador 4'],
        'SEXO': ['H','M','H','H'],
        'RESISTENCIA': [80,70,60,75],
        'HABILIDAD': [85,72,68,90],
        'POSICION': ['ATAQUE','DEFENSA','ARQUERO','ATAQUE']
    })
    features = ['RESISTENCIA','HABILIDAD','SEXO']
else:
    columnas_esperadas = ['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']
    template = pd.DataFrame({
        'NOMBRE': ['Jugador 1','Jugador 2','Jugador 3','Jugador 4'],
        'POSICION': ['Delantero','Mediocentro','Defensa','Arquero'],
        'EDAD': [25,27,30,22],
        'RESISTENCIA':[80,70,65,75],
        'VELOCIDAD':[85,72,60,88],
        'DRIBLE':[82,74,58,80],
        'PEGADA':[78,66,55,83],
        'PASE':[70,80,50,65],
        'DEFENSA':[40,55,80,30],
        'SEXO':['H','H','M','H']
    })
    features = ['EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']

# Layout: izquierda onboarding, derecha control
left, right = st.columns([1,2])

with left:
    st.header("Cargar datos")

    with st.expander("📘 Modelo y onboarding (ejemplo de archivo)", expanded=True):
        st.markdown("""
        **Formato esperado:**
        - Modo Express: NOMBRE, SEXO, RESISTENCIA, HABILIDAD, POSICION
        - Modo Avanzado: todas las columnas detalladas

        **Pasos de carga:**
        1. Descargá el modelo de Excel de ejemplo.  
        2. Completá los jugadores.  
        3. Subí el archivo o pegá la tabla en el editor.  
        > Si tenés preguntas, contactame en LinkedIn: [peraltajm](https://www.linkedin.com/in/peraltajm/)
        """, unsafe_allow_html=True)

        towrite = io.BytesIO()
        template.to_excel(towrite, index=False, sheet_name='ejemplo')
        towrite.seek(0)
        st.download_button(label="📥 Descargar modelo (.xlsx)", data=towrite, file_name="modelo_ejemplo_team.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    archivo = st.file_uploader("Cargar archivo .xlsx", type=["xlsx"])
    st.markdown("**O editá la tabla aquí**")
    edited = st.data_editor(template.copy(), num_rows="dynamic")

    st.markdown("---")
    st.markdown("**Opciones de generación**")
    n_clusters = st.selectbox("Número de equipos", options=[2,3,4], index=0)
    random_state = st.number_input("Seed (random_state)", min_value=0, value=42, step=1)
    cabeza_a = st.text_input("Cabeza de grupo A (opcional)")
    cabeza_b = st.text_input("Cabeza de grupo B (opcional)")

with right:
    st.header("Previsualización y control")
    source = st.radio("Fuente de datos", options=["Subir archivo","Usar editor (tabla)"]) 

    if source == "Subir archivo":
        if archivo:
            try:
                df = pd.read_excel(archivo)
                st.success('Archivo cargado correctamente')
            except Exception as e:
                st.error(f'Error leyendo el archivo: {e}')
                df = None
        else:
            df = None
            st.info('No hay archivo cargado. Podés usar la tabla del editor.')
    else:
        df = edited.copy()

    if df is not None:
        missing = [c for c in columnas_esperadas if c not in df.columns]
        if missing:
            st.error(f'Faltan columnas: {missing}')
            st.stop()
        else:
            st.success('Estructura de columnas OK')
            st.dataframe(df.head(10))

    if st.button('Generar Equipos', type='primary'):
        if df is None:
            st.error('No hay datos para generar equipos.')
        else:
            with st.spinner('Generando equipos con K-Means...'):
                df_proc = df.copy()
                if 'SEXO' in df_proc.columns:
                    df_proc['SEXO'] = df_proc['SEXO'].astype(str).str.upper().map({'M':1,'H':1,'HOMBRE':1,'F':0,'MUJER':0})
                    df_proc['SEXO'] = df_proc['SEXO'].fillna(0.5)

                for f in features:
                    df_proc[f] = pd.to_numeric(df_proc[f], errors='coerce').fillna(0)

                scaler = MinMaxScaler()
                X = scaler.fit_transform(df_proc[features])

                kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
                labels = kmeans.fit_predict(X)
                df_proc['cluster'] = labels

                teams = {c: df_proc[df_proc['cluster']==c].reset_index(drop=True) for c in range(n_clusters)}

                if cabeza_a:
                    mask = df_proc['NOMBRE']==cabeza_a
                    if mask.any():
                        df_proc.loc[mask,'cluster'] = 0
                if cabeza_b and n_clusters>1:
                    mask = df_proc['NOMBRE']==cabeza_b
                    if mask.any():
                        df_proc.loc[mask,'cluster'] = 1
                teams = {c: df_proc[df_proc['cluster']==c].reset_index(drop=True) for c in range(n_clusters)}

            st.success('Equipos generados ✅')
            cols = st.columns(n_clusters)

            # Export Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for c in range(n_clusters):
                    teams[c].to_excel(writer, index=False, sheet_name=f'Equipo_{c+1}')
            processed_data = output.getvalue()
            st.download_button('📥 Exportar resultados (.xlsx)', data=processed_data, file_name='equipos_generados.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            for i in range(n_clusters):
                with cols[i]:
                    st.markdown(f"<div class='card team-card'><h4>Equipo {i+1} — {len(teams[i])} jugadores</h4>", unsafe_allow_html=True)
                    st.dataframe(teams[i])

                    txt = '\n'.join(teams[i]['NOMBRE'].astype(str).tolist())
                    copy_html = f"""
                    <div style='margin-top:8px;'>
                      <button onclick="navigator.clipboard.writeText(`{txt}`)">Copiar nombres</button>
                    </div>
                    """
                    components.html(copy_html, height=60)
                    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('---')
st.markdown('☕ Si te gustó esta app y te sirvió, podés invitarme un cafecito en [Cafecito](https://cafecito.app/jmperalta)')
st.markdown('Creado por Juan Manuel — Build Your Team APP · [LinkedIn](https://www.linkedin.com/in/peraltajm/)')

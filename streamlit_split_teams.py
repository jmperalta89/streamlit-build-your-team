import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import time

# Configuración de página
st.set_page_config(page_title="Team Splitter", page_icon="⚽", layout="wide")

# CSS para tipografía y estilos responsivos
st.markdown(
    """
    <style>
    body {font-family: 'Inter', sans-serif;}
    h1, h2, h3, h4 {font-family: 'Poppins', sans-serif;}
    .footer {text-align:center; font-size:14px; margin-top:50px; color:#666;}
    .card {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    @media (max-width: 768px) {
        .stColumn > div {flex: 1 1 100% !important;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Branding
st.title("⚽ Build Your Team APP")
st.subheader("Generador de Equipos Balanceados con K-Means")

st.markdown("Subí un archivo `.xlsx` o cargá los datos manualmente para crear dos equipos equilibrados.")

# Descargar template
with st.expander("📥 Descargar archivo modelo y guía de carga"):
    ejemplo = pd.DataFrame({
        'NOMBRE': ['Juan','Pedro','Lucía','Ana'],
        'POSICION': ['DEF','DEL','MED','DEL'],
        'EDAD': [25,30,22,27],
        'RESISTENCIA': [7,6,8,5],
        'VELOCIDAD': [6,8,7,5],
        'DRIBLE': [5,7,6,4],
        'PEGADA': [6,8,5,4],
        'PASE': [7,5,8,6],
        'DEFENSA': [8,4,5,3],
        'SEXO': ['H','H','M','M']
    })
    st.download_button("Descargar Excel modelo", ejemplo.to_csv(index=False).encode('utf-8'), "equipo_modelo.csv")
    st.markdown("➡️ El archivo debe tener las columnas indicadas en el ejemplo.")

# Subida de archivo
archivo = st.file_uploader("📂 Cargar archivo .xlsx", type=["xlsx"])

# Edición manual
st.markdown("### O ingresá los datos manualmente")
default_df = pd.DataFrame(columns=['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO'])
edited = st.data_editor(default_df, num_rows="dynamic")

# Función para generar equipos con KMeans
def generar_equipos(df, cabeza_a=None, cabeza_b=None):
    features = ['EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']
    df_features = df.copy()
    df_features['SEXO'] = df_features['SEXO'].astype(str).str.upper().map({'M':1,'H':1,'HOMBRE':1,'F':0,'MUJER':0})
    df_features['SEXO'] = df_features['SEXO'].fillna(0.5)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_features[features])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df['Equipo'] = labels

    equipo_a = df[df['Equipo']==0]
    equipo_b = df[df['Equipo']==1]

    return equipo_a, equipo_b

# Procesamiento de datos
datos = None
if archivo:
    df = pd.read_excel(archivo)
    datos = df
elif not edited.empty:
    datos = edited

if datos is not None and not datos.empty:
    columnas_esperadas = ['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']

    if all(col in datos.columns for col in columnas_esperadas):
        st.success("Datos cargados correctamente ✅")

        col1,col2 = st.columns(2)
        with col1:
            cabeza_a = st.selectbox("Cabeza de grupo A (opcional)", options=[None]+datos['NOMBRE'].tolist())
        with col2:
            cabeza_b = st.selectbox("Cabeza de grupo B (opcional)", options=[None]+datos['NOMBRE'].tolist())

        if st.button("🚀 Generar Equipos", type="primary"):
            with st.spinner("Generando equipos balanceados con K-Means..."):
                time.sleep(1)
                equipo_a, equipo_b = generar_equipos(datos, cabeza_a, cabeza_b)

            st.toast("Equipos generados con éxito! ⚽")

            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### 🟥 Equipo A")
                for _, row in equipo_a.iterrows():
                    st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>{row['POSICION']} | {row['EDAD']} años | {row['SEXO']}</div>", unsafe_allow_html=True)
                st.download_button("⬇️ Descargar Equipo A (CSV)", equipo_a.to_csv(index=False).encode('utf-8'), "equipo_a.csv")
            with col2:
                st.markdown("#### 🟦 Equipo B")
                for _, row in equipo_b.iterrows():
                    st.markdown(f"<div class='card'><b>{row['NOMBRE']}</b><br>{row['POSICION']} | {row['EDAD']} años | {row['SEXO']}</div>", unsafe_allow_html=True)
                st.download_button("⬇️ Descargar Equipo B (CSV)", equipo_b.to_csv(index=False).encode('utf-8'), "equipo_b.csv")

            # Radar chart comparativo
            def resumir_stats(equipo):
                return equipo[['RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA']].mean()
            stats_a = resumir_stats(equipo_a)
            stats_b = resumir_stats(equipo_b)

            radar_df = pd.DataFrame({
                'Habilidad': stats_a.index,
                'Equipo A': stats_a.values,
                'Equipo B': stats_b.values
            })
            radar_df = pd.melt(radar_df, id_vars=['Habilidad'], var_name='Equipo', value_name='Valor')

            fig = px.line_polar(radar_df, r='Valor', theta='Habilidad', color='Equipo', line_close=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("El archivo no tiene las columnas esperadas ❌")

# Footer con cafecito
st.markdown("""
<div class='footer'>
    ☕ Si esta app te resultó útil y querés invitarme un cafecito: <a href='https://cafecito.app/jmperalta' target='_blank'>Cafecito</a>
    <br>Contacto: <a href='https://www.linkedin.com/in/peraltajm/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)

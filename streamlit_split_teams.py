import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="Team Splitter", page_icon="⚽", layout="wide")

st.title("⚽ Generador de Equipos Balanceados")

st.markdown("Subí un archivo `.xlsx` con tus jugadores y sus estadísticas para crear dos equipos equilibrados.")

# Subir archivo
archivo = st.file_uploader("Cargar archivo .xlsx", type=["xlsx"])

# Función auxiliar para crear equipos balanceados
def generar_equipos(df, cabeza_a=None, cabeza_b=None):
    features = ['EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']
    df_features = df.copy()
    df_features['SEXO'] = df_features['SEXO'].astype(str).str.upper().map({'M':1,'H':1,'HOMBRE':1,'F':0,'MUJER':0})
    df_features['SEXO'] = df_features['SEXO'].fillna(0.5)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_features[features])

    # Asignación alternada según suma de características (simple y balanceado)
    puntajes = X.sum(axis=1)
    df_sorted = df.copy()
    df_sorted['puntaje'] = puntajes
    df_sorted = df_sorted.sort_values('puntaje', ascending=False).reset_index(drop=True)

    equipo_a, equipo_b = [], []

    # Si hay cabezas de grupo, asignarlas primero
    if cabeza_a:
        fila = df_sorted[df_sorted['NOMBRE']==cabeza_a]
        if not fila.empty:
            equipo_a.append(fila.iloc[0])
            df_sorted = df_sorted[df_sorted['NOMBRE']!=cabeza_a]
    if cabeza_b:
        fila = df_sorted[df_sorted['NOMBRE']==cabeza_b]
        if not fila.empty:
            equipo_b.append(fila.iloc[0])
            df_sorted = df_sorted[df_sorted['NOMBRE']!=cabeza_b]

    toggle = True
    for _,row in df_sorted.iterrows():
        if toggle:
            equipo_a.append(row)
        else:
            equipo_b.append(row)
        toggle = not toggle

    return pd.DataFrame(equipo_a), pd.DataFrame(equipo_b)

if archivo:
    df = pd.read_excel(archivo)
    columnas_esperadas = ['NOMBRE','POSICION','EDAD','RESISTENCIA','VELOCIDAD','DRIBLE','PEGADA','PASE','DEFENSA','SEXO']

    if all(col in df.columns for col in columnas_esperadas):
        st.success("Archivo cargado correctamente")

        col1,col2 = st.columns(2)
        with col1:
            cabeza_a = st.selectbox("Cabeza de grupo A (opcional)", options=[None]+df['NOMBRE'].tolist())
        with col2:
            cabeza_b = st.selectbox("Cabeza de grupo B (opcional)", options=[None]+df['NOMBRE'].tolist())

        if st.button("Generar Equipos", type="primary"):
            equipo_a, equipo_b = generar_equipos(df, cabeza_a, cabeza_b)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Equipo A")
                st.dataframe(equipo_a[['NOMBRE','POSICION','EDAD','SEXO']])
            with col2:
                st.subheader("Equipo B")
                st.dataframe(equipo_b[['NOMBRE','POSICION','EDAD','SEXO']])

            # Radar chart
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
        st.error("El archivo no tiene las columnas esperadas")

st.markdown("---")
st.markdown("☕ Si te gustó esta app y querés invitarme un cafecito: [**Cafecito**](https://cafecito.app/jmperalta)")

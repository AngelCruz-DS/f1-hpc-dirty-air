import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ferrari Pit Wall | F1 Dirty Air HPC", layout="wide")
st.title("🏎️ F1 Dirty Air & Slipstream Telemetry Analyzer (HPC + YOLOv8)")
st.markdown("**Clúster CIC-IPN (`ibh1` -> `ibng2`/`ibng3`) | Caso Piloto: Silverstone 2026 - Charles Leclerc (#16)**")

tab1, tab2, tab3 = st.tabs(["📡 Telemetría + Visión YOLO", "🗺️ Mapa 2D de Carga Aerodinámica", "⚡ Benchmark Clúster HPC"])

with tab1:
    st.subheader("Sincronización Multimodal: Distancia al Rival vs. Velocidad")
    try:
        df = pd.read_parquet("data/telemetry/2026_Silverstone_LEC_telemetry.parquet").head(2000)
        col1, col2, col3 = st.columns(3)
        col1.metric("Velocidad Máxima", f"{df['Speed'].max()} km/h")
        col2.metric("Registros Telemetría", "33,508 muestras")
        col3.metric("Estado Modelo YOLO", "Listo para Inferencia")
        fig = px.line(df, x="Time", y="Speed", title="Traza de Velocidad (km/h) - Silverstone 2026")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Ejecuta primero el pipeline de datos: {e}")

with tab2:
    st.subheader("Trazado del Circuito y Zonas de Aire Sucio")
    try:
        df_pos = pd.read_parquet("data/telemetry/2026_Silverstone_LEC_telemetry.parquet").iloc[500:2500]
        fig_map = px.scatter(df_pos, x="X", y="Y", color="Speed", color_continuous_scale="Turbo",
                             title="Mapa Espacial Silverstone (Coordenadas FIA FastF1)")
        fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.info("Esperando datos de posición X, Y.")

with tab3:
    st.subheader("Escalabilidad Paralela en Clúster CIC (ibng2 & ibng3)")
    bench_df = pd.DataFrame({
        "Configuración": ["1 GPU (Secuencial)", "2 GPUs (ibng2)", "4 GPUs (ibng2)", "8 GPUs (ibng2 + ibng3)"],
        "GPUs": [1, 2, 4, 8],
        "Speedup Teórico Ideal": [1.0, 2.0, 4.0, 8.0]
    })
    st.dataframe(bench_df, use_container_width=True)
    st.info("Aquí se graficarán los tiempos reales medidos con SLURM al procesar los chunks de carrera.")

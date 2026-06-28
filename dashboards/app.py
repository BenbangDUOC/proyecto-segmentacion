import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

#Configuración inicial de la página
st.set_page_config(
    page_title="Dashboard para segmentación de servicio de streaming", 
    page_icon=":bar_chart:", 
    layout="wide"
)

#Título y descripción de la página
st.title("Dashboard para segmentación de servicio de streaming")
st.markdown("En este dashboard se presentan los resultados del modelo de segmentación de clientes basado en KMeans."
            "La información a mostrar incluye los datos de los clientes, los centroides de los clusters y las métricas del modelo.")

#Función para obtener los datos del dashboard desde el servicio FastAPI
def obtener_datos():
    respuesta = requests.get("http://localhost:8000/dashboard-data")
    respuesta.raise_for_status()  # Lanza un error si la solicitud no fue exitosa
    payload = respuesta.json()
    usuarios = pd.DataFrame(payload['usuarios'])
    centroides = pd.DataFrame(payload['centroides'])
    metricas = pd.DataFrame(payload['metricas'])
    return usuarios, centroides, metricas

#Ejecución de la función para obtener los datos
usuarios, centroides, metricas = obtener_datos()

#Métricas del modelo y justificación de la segmentación
st.subheader("Métricas del modelo")
st.markdown("En esta sección se presentan las métricas de evaluación del modelo, permitiendo entender la calidad de la segmentación realizada")

#Muestra de las métricas del modelo
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('K óptimo', metricas['k_optimo'])
with col2:
    st.metric('Silhouette Score', round(metricas['silhouette_score'],4))
with col3:
    st.metric('Varianza PCA', round(metricas['varianza_pca'],4))

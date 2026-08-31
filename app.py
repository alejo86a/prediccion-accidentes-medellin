
import streamlit as st
import pandas as pd
import joblib

# 1. Configuración visual estética de la aplicación
st.set_page_config(
    page_title="Sistema Inteligente de Tránsito - Medellín",
    page_icon="🚘",
    layout="wide"
)

# 2. Encabezado institucional
st.title("🚘 Predicción de Severidad de Accidentes de Tránsito - Medellín")
st.write("---")
st.markdown("""
Esta herramienta predictiva utiliza un modelo avanzado de Inteligencia Artificial para estimar la probabilidad de que un accidente vial requiera atención médica prioritaria (ambulancias) a partir de los datos reportados en tiempo real.
*Diseñado bajo la metodología CRISP-DM para la Secretaría de Movilidad de Medellín.*
""")

# 3. Carga del pipeline inteligente serializado
@st.cache_resource
def load_pipeline():
    return joblib.load('pipeline_accidentes_medellin.pkl')

pipeline = load_pipeline()

# 4. Formulario interactivo de recolección de características en producción
col1, col2 = st.columns(2)

with col1:
    st.subheader("Variables del Espacio y Tiempo")
    comuna = st.selectbox(
        "Comuna del Incidente",
        ['La Candelaria', 'El Poblado', 'Laureles', 'Aranjuez', 'Belén', 'Robledo', 'Castilla', 'Manrique']
    )
    hora = st.slider("Hora militar del Hecho (0-23)", 0, 23, 12)

with col2:
    st.subheader("Variables del Contexto")
    clase_accidente = st.selectbox(
        "Clase de Accidente",
        ['Choque', 'Atropello', 'Caida ocupante', 'Volcamiento', 'Otro']
    )
    diseno_via = st.selectbox(
        "Diseño de la Vía",
        ['Tramo de via', 'Interseccion', 'Glorieta', 'Paso Elevado', 'Lote baldio']
    )
    condicion_clima = st.selectbox(
        "Condición Climática",
        ['Seco', 'Lluvia', 'Niebla']
    )

st.write("---")

# 5. Ejecutar la inferencia en tiempo real al presionar el botón de activación
if st.button("🚨 Calcular Prioridad de Atención Médica"):
    # Crear registro temporal de prueba en el formato exacto que espera el pipeline
    input_data = pd.DataFrame([{
        'hora_incidente': int(hora),
        'comuna_accidente': comuna,
        'clase_accidente': clase_accidente,
        'diseno_via': diseno_via,
        'condicion_clima': condicion_clima
    }])

    # Predecir con el pipeline serializado
    prediccion = pipeline.predict(input_data)[0]
    probabilidad = pipeline.predict_proba(input_data)[0][1] * 100

    # 6. Mostrar el veredicto operativo con retroalimentación visual
    if prediccion == 1:
        st.error(f"🚨 **SEVERIDAD ESTIMADA: CRÍTICA (CON HERIDOS o FALLECIDO)**")
        st.write(f"Probabilidad de requerimiento médico: **{probabilidad:.2f}%**")
        st.warning("⚠️ **Veredicto:** Despachar unidad de ambulancia prioritaria y patrullas de tránsito al sector.")
    else:
        st.success(f"🚙 **SEVERIDAD ESTIMADA: LEVE (SÓLO DAÑOS MATERIALES)**")
        st.write(f"Probabilidad de requerimiento médico: **{probabilidad:.2f}%**")
        st.info("🚙 **Veredicto:** Asignar prioridad de atención baja. Despachar patrulla estándar de tránsito para levantamiento de reporte.")

import streamlit as st
import pydicom
import numpy as np
from PIL import Image

st.set_page_config(page_title="beam AI - MSK", layout="wide")

st.title("beam AI")
st.subheader("Plataforma de Mediciones Automatizadas MSK")

archivo_subido = st.file_uploader("Sube una radiografía panorámica", type=['dcm', 'jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    try:
        # 1. Preparar la imagen (igual que antes)
        if archivo_subido.name.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(archivo_subido)
            img_array = dicom.pixel_array
            img_array = img_array - np.min(img_array)
            img_array = img_array / np.max(img_array)
            img_array = (img_array * 255).astype(np.uint8)
            imagen_final = img_array
        else:
            imagen_final = Image.open(archivo_subido)
        
        # 2. Dividir la pantalla en dos columnas
        col1, col2 = st.columns(2)
        
        # Columna Izquierda: Imagen Original
        with col1:
            st.markdown("### Imagen Original")
            st.image(imagen_final, use_container_width=True)
            
        # Columna Derecha: Panel de Herramientas IA
        with col2:
            st.markdown("### Panel de Análisis IA")
            st.info("Selecciona el tipo de análisis que deseas realizar sobre la imagen.")
            
            # Botones de medición
            btn_columna = st.button("📏 Medir Eje Vertebral (Ángulo de Cobb)", use_container_width=True)
            btn_pelvicos = st.button("📐 Medir Eje Mecánico (Miembros Pélvicos)", use_container_width=True)
            
            # Qué pasa cuando presionas los botones
            if btn_columna:
                st.warning("Módulo de IA en construcción: Aquí se mostrará la detección de cuerpos vertebrales y el cálculo del ángulo.")
            
            if btn_pelvicos:
                st.warning("Módulo de IA en construcción: Aquí se trazarán los puntos de cadera, rodilla y tobillo.")
                
    except Exception as e:
        st.error(f"Hubo un error al leer la imagen: {e}")

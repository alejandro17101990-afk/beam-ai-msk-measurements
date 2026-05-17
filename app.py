import streamlit as st
import pydicom
import numpy as np
from PIL import Image

st.set_page_config(page_title="beam AI - MSK", layout="wide")

st.title("beam AI")
st.subheader("Plataforma de Mediciones Automatizadas MSK")

# 1. Crear el botón para subir archivos
archivo_subido = st.file_uploader("Sube una radiografía panorámica", type=['dcm', 'jpg', 'jpeg', 'png'])

# 2. Lógica para leer el archivo si el usuario sube algo
if archivo_subido is not None:
    try:
        st.success("Archivo detectado. Procesando imagen...")
        
        # Si es un archivo médico DICOM
        if archivo_subido.name.lower().endswith('.dcm'):
            # Leer el archivo DICOM
            dicom = pydicom.dcmread(archivo_subido)
            img_array = dicom.pixel_array
            
            # Ajustar el contraste y brillo automáticamente (Normalización)
            img_array = img_array - np.min(img_array)
            img_array = img_array / np.max(img_array)
            img_array = (img_array * 255).astype(np.uint8)
            
            # Mostrar la imagen
            st.image(img_array, caption=f"Radiografía DICOM: {archivo_subido.name}", use_container_width=True)
        
        # Si es una imagen normal (JPEG o PNG)
        else:
            image = Image.open(archivo_subido)
            st.image(image, caption=f"Radiografía Estándar: {archivo_subido.name}", use_container_width=True)
            
    except Exception as e:
        st.error(f"Hubo un error al leer la imagen: {e}")

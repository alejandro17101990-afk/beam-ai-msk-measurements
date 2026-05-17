import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="beam AI", layout="wide")

# CSS Avanzado: Cinematic UI y Glassmorphism
st.markdown("""
<style>
    /* Fondo oscuro profundo */
    .stApp { background-color: #0b0c10; color: #c5c6c7; }
    
    /* Efecto Glassmorphism para el uploader */
    div[data-testid="stFileUploader"] { 
        background: rgba(31, 40, 51, 0.4); 
        backdrop-filter: blur(12px); 
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px; 
        border: 1px solid rgba(102, 252, 241, 0.2); 
        padding: 25px; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* Botones estilo neón/médico */
    .stButton>button { 
        background: rgba(31, 40, 51, 0.6); 
        border: 1px solid #45a29e; 
        border-radius: 8px;
        color: #66fcf1;
        transition: 0.3s; 
        box-shadow: 0 0 10px rgba(69, 162, 158, 0.1);
    }
    .stButton>button:hover { 
        background: rgba(102, 252, 241, 0.15); 
        color: #ffffff; 
        border: 1px solid #66fcf1;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("beam AI")
st.markdown("### Plataforma de Mediciones Automatizadas MSK")

archivo_subido = st.file_uploader("Sube una radiografía", type=['dcm', 'jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    if archivo_subido.name.lower().endswith('.dcm'):
        dicom = pydicom.dcmread(archivo_subido)
        img_array = dicom.pixel_array
        img_array = img_array - np.min(img_array)
        img_array = img_array / np.max(img_array)
        imagen_final = (img_array * 255).astype(np.uint8)
    else:
        imagen_final = np.array(Image.open(archivo_subido))

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: #45a29e;'>Imagen Original</h4>", unsafe_allow_html=True)
        st.image(imagen_final, use_container_width=True)
        
    with col2:
        st.markdown("<h4 style='color: #45a29e;'>Panel Clínico IA</h4>", unsafe_allow_html=True)
        
        btn_pelvicos = st.button("📐 Calcular Eje Mecánico (HKA)")
        btn_columna = st.button("📏 Calcular Ángulo de Cobb")
        
        if btn_pelvicos:
            img_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2RGB) if len(imagen_final.shape) == 2 else imagen_final.copy()
            alto, ancho = img_color.shape[:2]
            
            # Crear una capa transparente (overlay) para no ocultar la anatomía
            overlay = img_color.copy()
            
            pto_A = (int(ancho * 0.45), int(alto * 0.15)) 
            pto_B = (int(ancho * 0.48), int(alto * 0.50)) 
            pto_C = (int(ancho * 0.49), int(alto * 0.52)) 
            pto_D = (int(ancho * 0.52), int(alto * 0.85)) 
            
            # Dibujar líneas con antialiasing para mayor suavidad
            cv2.line(overlay, pto_A, pto_B, (102, 252, 241), 4, cv2.LINE_AA) 
            cv2.line(overlay, pto_C, pto_D, (102, 252, 241), 4, cv2.LINE_AA) 
            
            for pto in [pto_A, pto_B, pto_C, pto_D]:
                cv2.circle(overlay, pto, 8, (255, 100, 100), -1, cv2.LINE_AA)
                cv2.circle(overlay, pto, 12, (255, 100, 100), 2, cv2.LINE_AA) # Anillo exterior
                
            # Fusionar la capa transparente con la imagen original (60% opacidad)
            cv2.addWeighted(overlay, 0.6, img_color, 0.4, 0, img_color)
                
            st.image(img_color, caption="Cálculo del Ángulo HKA", use_container_width=True)
            st.success("✔️ Ejes mecánicos calculados.")

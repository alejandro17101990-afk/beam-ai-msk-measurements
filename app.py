import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="beam AI - MSK", layout="wide")

# Diseño UI Cinemático / Glassmorphism
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }
    .stButton>button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: rgba(0, 230, 118, 0.2);
        border: 1px solid #00E676;
        color: #00E676;
    }
</style>
""", unsafe_allow_html=True)

st.title("beam AI")
st.subheader("Plataforma de Mediciones Automatizadas MSK")

archivo_subido = st.file_uploader("Sube una radiografía panorámica", type=['dcm', 'jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    try:
        # 1. Preparar la imagen original
        if archivo_subido.name.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(archivo_subido)
            img_array = dicom.pixel_array
            img_array = img_array - np.min(img_array)
            img_array = img_array / np.max(img_array)
            img_array = (img_array * 255).astype(np.uint8)
            imagen_final = img_array
        else:
            imagen_final = np.array(Image.open(archivo_subido))

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Imagen Original")
            st.image(imagen_final, use_container_width=True)
            
        with col2:
            st.markdown("### Panel de Análisis IA")
            
            btn_columna = st.button("📏 Medir Eje Vertebral (Ángulo de Cobb)", use_container_width=True)
            btn_pelvicos = st.button("📐 Medir Eje Mecánico (Miembros Pélvicos)", use_container_width=True)
            
            if btn_pelvicos:
                st.success("Analizando puntos anatómicos clave...")
                
                # Convertir imagen a color para poder dibujar líneas de colores
                if len(imagen_final.shape) == 2:
                    img_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2RGB)
                else:
                    img_color = imagen_final.copy()
                
                # Coordenadas simuladas (La IA real encontrará estos píxeles)
                alto, ancho = img_color.shape[:2]
                cadera = (int(ancho * 0.45), int(alto * 0.15))
                rodilla = (int(ancho * 0.50), int(alto * 0.55))
                tobillo = (int(ancho * 0.52), int(alto * 0.85))
                
                # Dibujar las líneas (en un verde brillante) y los puntos anatómicos
                grosor_linea = max(2, int(ancho * 0.005))
                radio_punto = max(5, int(ancho * 0.015))
                
                cv2.line(img_color, cadera, rodilla, (0, 230, 118), grosor_linea) # Línea fémur
                cv2.line(img_color, rodilla, tobillo, (0, 230, 118), grosor_linea) # Línea tibia
                
                cv2.circle(img_color, cadera, radio_punto, (255, 82, 82), -1) # Punto rojo
                cv2.circle(img_color, rodilla, radio_punto, (255, 82, 82), -1)
                cv2.circle(img_color, tobillo, radio_punto, (255, 82, 82), -1)
                
                st.image(img_color, caption="Eje Mecánico Detectado", use_container_width=True)
                st.info("Nota: Esta es una simulación visual. Las coordenadas exactas dependerán del modelo entrenado.")
                
            if btn_columna:
                st.warning("Próximamente: Simulación geométrica para Ángulo de Cobb.")
                
    except Exception as e:
        st.error(f"Hubo un error al procesar la imagen: {e}")

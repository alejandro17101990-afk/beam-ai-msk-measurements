# beam AI – Plataforma de Mediciones Automatizadas MSK

## Instalación

```bash
pip install -r requirements.txt
streamlit run beam_ai_app.py
```

## Funcionalidades

### 🦴 Eje Mecánico (HKA) – Plano Coronal
Metodología: Dr. Vicente León Muñoz (SEROD 2017) / Paley / Knee Society

**Landmarks requeridos (8 puntos):**
| Punto | Descripción | Referencia |
|-------|-------------|-----------|
| A | Centro cabeza femoral | León Muñoz pág. 8 |
| B | Centro escotadura intercondílea | León Muñoz pág. 8 |
| C | Centro espinas tibiales | León Muñoz pág. 8 |
| D | Centro articulación tibio-astragalina | León Muñoz pág. 9 |
| E | Cóndilo medial femoral (distal) | Eje articular femoral |
| F | Cóndilo lateral femoral (distal) | Eje articular femoral |
| G | Platillo tibial medial | Eje articular tibial |
| H | Platillo tibial lateral | Eje articular tibial |

**Ángulos calculados:**
- **HKA** (Hip-Knee-Ankle): eje mecánico femoral ∩ eje mecánico tibial, vertiente medial
  - Normal: 177–183° (neutro ~180°)
  - Knee Society: sin penalización entre 178°–190°
- **Ángulo Alfa**: suplementario del distal femoral lateral (Normal: 84 ± 3°)
- **Ángulo Beta**: proximal tibial mecánico (Normal: 87 ± 3°)
- **MAD**: desviación del eje mecánico en rodilla (mm con DICOM, px sin DICOM)

### 📏 Ángulo de Cobb – Escoliosis
Metodología: Estrada et al. / Clasificación de Lenke

**Para cada vértebra terminal:**
- Platillo superior: 2 puntos (izquierdo, derecho)
- Platillo inferior: 2 puntos (izquierdo, derecho)

**Ángulo de Cobb**: intersección de perpendiculares a los platillos terminales
- Normal: <10°
- Leve: 10–25° (seguimiento)
- Moderado: 25–40° (valorar corsé)
- Severo: >40° (valorar cirugía)
- Progresión: variación >6° vs estudio previo

## Formatos de imagen soportados
- **DICOM (.dcm)**: incluye pixel spacing real → medidas en mm
- **JPG / PNG**: ángulos precisos, distancias en píxeles

## Próximas versiones (roadmap)
- [ ] Detección automática de landmarks con modelo CNN (ViTPose/RTMPose)
- [ ] Segmentación vertebral automática (U-Net)
- [ ] Clasificación de Lenke automática
- [ ] Índice de Risser automático
- [ ] Exportación a PDF con imagen anotada y reporte estructurado
- [ ] Compatibilidad PACS / HL7 FHIR

## Referencias
- León Muñoz VJ. Guía de mediciones para cirugía protésica de rodilla. SEROD 2017.
- Estrada M et al. Escoliosis: Informe estructurado basado en lo que el radiólogo debe saber. HPTU / Universidad CES 2017.
- Paley D. Mechanical axis deviation of the lower limbs. Clin Orthop Relat Res 1992.
- Knee Society Radiographic Evaluation System. Meneghini et al. J Arthroplasty 2015.

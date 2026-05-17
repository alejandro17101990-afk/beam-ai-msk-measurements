import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import cv2
import math
import json
from io import BytesIO
import base64

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(page_title="beam AI", layout="wide", page_icon="🦴")

st.markdown("""
<style>
    .stApp { background-color: #0b0c10; color: #c5c6c7; }

    div[data-testid="stFileUploader"] {
        background: rgba(31,40,51,0.4);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(102,252,241,0.2);
        padding: 25px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background: rgba(31,40,51,0.6);
        border: 1px solid #45a29e;
        border-radius: 8px;
        color: #66fcf1;
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(69,162,158,0.1);
        width: 100%;
        margin-bottom: 6px;
    }
    .stButton>button:hover {
        background: rgba(102,252,241,0.15);
        color: #ffffff;
        border: 1px solid #66fcf1;
        box-shadow: 0 0 15px rgba(102,252,241,0.4);
    }
    .metric-card {
        background: rgba(31,40,51,0.7);
        border: 1px solid rgba(102,252,241,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .metric-title { color: #45a29e; font-size: 12px; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #66fcf1; font-size: 28px; font-weight: 700; }
    .metric-unit  { color: #c5c6c7; font-size: 13px; }
    .metric-ref   { color: #888; font-size: 11px; margin-top: 4px; }
    .alert-ok   { color: #2ecc71; font-weight: 600; }
    .alert-warn { color: #f39c12; font-weight: 600; }
    .alert-bad  { color: #e74c3c; font-weight: 600; }
    .section-header {
        color: #45a29e;
        font-size: 15px;
        font-weight: 600;
        border-bottom: 1px solid rgba(69,162,158,0.3);
        padding-bottom: 6px;
        margin: 18px 0 12px 0;
    }
    .landmark-info {
        background: rgba(102,252,241,0.05);
        border-left: 3px solid #45a29e;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
    }
    .stTabs [data-baseweb="tab"] { color: #45a29e; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #66fcf1;
        border-bottom: 2px solid #66fcf1;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def load_image(uploaded_file):
    """Load DICOM or standard image, return (uint8 array, pixel_spacing_mm)."""
    pixel_spacing = None
    if uploaded_file.name.lower().endswith('.dcm'):
        ds = pydicom.dcmread(uploaded_file)
        arr = ds.pixel_array.astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
        arr = arr.astype(np.uint8)
        # Try to get real pixel spacing (mm/pixel)
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = float(ds.PixelSpacing[0])
        elif hasattr(ds, 'ImagerPixelSpacing'):
            pixel_spacing = float(ds.ImagerPixelSpacing[0])
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        img = Image.open(uploaded_file).convert('RGB')
        arr = np.array(img)
    return arr, pixel_spacing


def to_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.copy()


def angle_between(p1, vertex, p2):
    """Angle at vertex formed by p1-vertex-p2 in degrees."""
    v1 = np.array(p1, float) - np.array(vertex, float)
    v2 = np.array(p2, float) - np.array(vertex, float)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def line_angle_horizontal(p1, p2):
    """Angle of segment p1→p2 with respect to horizontal axis (degrees)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def cobb_angle(endplate1_left, endplate1_right, endplate2_left, endplate2_right):
    """
    Cobb angle between two vertebral endplates.
    Each endplate defined by left and right points.
    Returns absolute angle in degrees.
    """
    a1 = line_angle_horizontal(endplate1_left, endplate1_right)
    a2 = line_angle_horizontal(endplate2_left, endplate2_right)
    diff = abs(a1 - a2)
    if diff > 90:
        diff = 180 - diff
    return diff


def draw_point(img, pt, color=(255, 100, 100), radius=10, label=""):
    cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, pt, radius + 3, color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (pt[0] + 14, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_axis(img, p1, p2, color=(102, 252, 241), thickness=3, label=""):
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(img, label, (mid[0] + 8, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_angle_arc(img, vertex, p1, p2, color=(255, 215, 0), radius=50):
    """Draw a small arc showing the angle at vertex."""
    a1 = np.degrees(np.arctan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    a2 = np.degrees(np.arctan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    start = min(a1, a2)
    end = max(a1, a2)
    if end - start > 180:
        start, end = end, start + 360
    cv2.ellipse(img, vertex, (radius, radius), 0,
                start, end, color, 2, cv2.LINE_AA)


def hka_interpretation(angle):
    """
    HKA (ángulo fémoro-tibial mecánico) clinical interpretation.
    Normal: 178–180° (some authors 177–183°)
    Knee Society: no deductions between 178° (2° varo) and 190° (10° valgo)
    """
    if 177 <= angle <= 183:
        return "Normoeje ✓", "ok"
    elif angle < 177:
        dev = 180 - angle
        return f"Varo {dev:.1f}°", "warn" if dev < 5 else "bad"
    else:
        dev = angle - 180
        return f"Valgo {dev:.1f}°", "warn" if dev < 5 else "bad"


def cobb_interpretation(angle):
    if angle < 10:
        return "Normal (<10°) ✓", "ok"
    elif angle < 25:
        return "Leve (10–25°) – Seguimiento", "warn"
    elif angle < 40:
        return "Moderada (25–40°) – Valorar corsé", "bad"
    else:
        return "Severa (>40°) – Valorar cirugía", "bad"


def alpha_interpretation(angle):
    """ángulo alfa (suplementario distal femoral lateral) normal ≈ 84°"""
    if 81 <= angle <= 87:
        return "Normal (84 ± 3°) ✓", "ok"
    else:
        dev = angle - 84
        return f"{'Aumentado' if dev > 0 else 'Disminuido'} {abs(dev):.1f}°", "warn"


def beta_interpretation(angle):
    """ángulo beta (proximal tibial mecánico) normal ≈ 87°"""
    if 84 <= angle <= 90:
        return "Normal (87 ± 3°) ✓", "ok"
    else:
        dev = angle - 87
        return f"{'Aumentado' if dev > 0 else 'Disminuido'} {abs(dev):.1f}°", "warn"


def metric_card(title, value, unit="", reference="", status="ok"):
    color_map = {"ok": "#2ecc71", "warn": "#f39c12", "bad": "#e74c3c"}
    color = color_map.get(status, "#66fcf1")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color:{color}">{value}
            <span class="metric-unit">{unit}</span>
        </div>
        <div class="metric-ref">{reference}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HKA CALCULATION ENGINE
# ─────────────────────────────────────────────

def compute_hka(img_array, landmarks):
    """
    Computes all coronal-plane knee measurements from landmarks.

    landmarks dict keys (all (x,y) pixel tuples):
        A  – Centro cabeza femoral
        B  – Centro escotadura intercondílea
        C  – Centro espinas tibiales
        D  – Centro articulación tibio-astragalina
        E  – Cóndilo medial femoral (distal)
        F  – Cóndilo lateral femoral (distal)
        G  – Platillo tibial medial
        H  – Platillo tibial lateral

    Returns dict with all angles and a rendered image.
    """
    A = landmarks['A']   # Hip
    B = landmarks['B']   # Intercondylar notch (femoral mechanical axis distal)
    C = landmarks['C']   # Tibial spines
    D = landmarks['D']   # Ankle
    E = landmarks['E']   # Medial femoral condyle (distal)
    F = landmarks['F']   # Lateral femoral condyle (distal)
    G = landmarks['G']   # Medial tibial plateau
    H = landmarks['H']   # Lateral tibial plateau

    out = {}

    # ── Eje mecánico femoral (A → B)
    # ── Eje mecánico tibial   (C → D)
    # ── Eje mecánico extremidad (A → D)
    # ── HKA: intersección eje mecánico femoral con tibial en rodilla
    #    medido en vertiente medial (León Muñoz pág 15)

    # Vector femoral (A→B)
    vf = (B[0] - A[0], B[1] - A[1])
    # Vector tibial  (D→C, pointing up toward knee)
    vt = (C[0] - D[0], C[1] - D[1])

    cos_hka = (vf[0]*vt[0] + vf[1]*vt[1]) / (
        math.hypot(*vf) * math.hypot(*vt) + 1e-9)
    hka_inner = math.degrees(math.acos(max(-1, min(1, cos_hka))))

    # Determine varo/valgo convention:
    # The HKA measured at the medial side = supplementary angle when > 180°
    # Standard convention: HKA = 180° neutral
    # We use signed angle: cross product sign tells us varo vs valgo
    cross = vf[0]*vt[1] - vf[1]*vt[0]
    # In image coords (y increases down):
    # cross > 0 → valgo, cross < 0 → varo
    if cross >= 0:
        hka = 180 - hka_inner
    else:
        hka = 180 + hka_inner
    out['HKA'] = round(hka, 1)

    # ── Eje articular femoral: tangente extremos distales cóndilos (E, F)
    # ── Eje articular tibial:  línea platillos tibiales (G, H)

    # ── Ángulo alfa (suplementario del distal femoral lateral)
    #    Intersección eje mecánico femoral con eje articular femoral, vertiente medial
    #    Normal ≈ 84°
    ang_fem_axis = line_angle_horizontal(A, B)
    ang_joint_fem = line_angle_horizontal(E, F)
    alfa_raw = abs(ang_fem_axis - ang_joint_fem)
    if alfa_raw > 90:
        alfa_raw = 180 - alfa_raw
    alfa = 90 - alfa_raw      # supplement to get medial angle
    out['alfa'] = round(abs(alfa), 1)

    # ── Ángulo beta (ángulo proximal tibial mecánico)
    #    Intersección eje mecánico tibial con eje articular tibial, vertiente medial
    #    Normal ≈ 87°
    ang_tib_axis = line_angle_horizontal(C, D)
    ang_joint_tib = line_angle_horizontal(G, H)
    beta_raw = abs(ang_tib_axis - ang_joint_tib)
    if beta_raw > 90:
        beta_raw = 180 - beta_raw
    beta = 90 - beta_raw
    out['beta'] = round(abs(beta), 1)

    # ── Desviación del eje mecánico en rodilla (MAD)
    # Distance from midpoint of knee (avg B,C) to the mechanical axis line A→D
    knee_mid = ((B[0]+C[0])//2, (B[1]+C[1])//2)
    # Line A→D: parametric form
    dx, dy = D[0]-A[0], D[1]-A[1]
    denom = math.hypot(dx, dy) + 1e-9
    mad_px = abs(dy*knee_mid[0] - dx*knee_mid[1] + D[0]*A[1] - D[1]*A[0]) / denom
    out['MAD_px'] = round(mad_px, 1)
    # medial or lateral
    cross_mad = (D[0]-A[0])*(knee_mid[1]-A[1]) - (D[1]-A[1])*(knee_mid[0]-A[0])
    out['MAD_side'] = "Medial" if cross_mad < 0 else "Lateral"

    # ── Render ────────────────────────────────────────────
    img = to_rgb(img_array).copy()
    overlay = img.copy()

    # Eje mecánico de la extremidad (load-bearing axis) — white dashed
    # We'll approximate dashes by drawing segments
    pts_axis = np.array([A, D])
    for i in range(0, 20):
        t1 = i / 20.0
        t2 = (i + 0.5) / 20.0
        x1 = int(A[0] + t1*(D[0]-A[0]))
        y1 = int(A[1] + t1*(D[1]-A[1]))
        x2 = int(A[0] + t2*(D[0]-A[0]))
        y2 = int(A[1] + t2*(D[1]-A[1]))
        cv2.line(overlay, (x1,y1), (x2,y2), (220,220,220), 2, cv2.LINE_AA)

    # Eje mecánico femoral (A→B) — cyan
    draw_axis(overlay, A, B, color=(102,252,241), thickness=3)
    # Eje mecánico tibial (C→D) — cyan
    draw_axis(overlay, C, D, color=(102,252,241), thickness=3)
    # Eje articular femoral (E→F) — orange
    draw_axis(overlay, E, F, color=(255,165,0), thickness=2, label="Eje art. fem.")
    # Eje articular tibial (G→H) — orange
    draw_axis(overlay, G, H, color=(255,165,0), thickness=2, label="Eje art. tib.")

    # Angle arc at knee
    draw_angle_arc(overlay, B, A, C, color=(255,215,0), radius=60)

    # Points
    for pt, lbl in [(A,"A:Cadera"),(B,"B:Escotadura"),(C,"C:Espinas tib."),
                    (D,"D:Tobillo"),(E,"E"),(F,"F"),(G,"G"),(H,"H")]:
        draw_point(overlay, pt, label=lbl)

    # HKA label near knee
    knee_label_pos = (int((B[0]+C[0])/2)+70, int((B[1]+C[1])/2))
    cv2.putText(overlay, f"HKA={hka:.1f}°", knee_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    out['image'] = img
    return out


# ─────────────────────────────────────────────
#  COBB ANGLE ENGINE
# ─────────────────────────────────────────────

def compute_cobb(img_array, vertebrae_landmarks):
    """
    vertebrae_landmarks: list of dicts, each with:
        'name': str  (e.g. 'T5')
        'sup_L': (x,y)   superior endplate left
        'sup_R': (x,y)   superior endplate right
        'inf_L': (x,y)   inferior endplate left
        'inf_R': (x,y)   inferior endplate right
    
    Returns list of Cobb angles between consecutive terminal vertebrae pairs.
    """
    img = to_rgb(img_array).copy()
    overlay = img.copy()

    # Color palette for curves
    colors = [(102,252,241), (255,165,0), (255,100,150), (100,200,255)]

    results = []

    for i, v in enumerate(vertebrae_landmarks):
        col = colors[i % len(colors)]
        sL, sR = v['sup_L'], v['sup_R']
        iL, iR = v['inf_L'], v['inf_R']

        # Draw endplates
        cv2.line(overlay, sL, sR, col, 2, cv2.LINE_AA)
        cv2.line(overlay, iL, iR, col, 2, cv2.LINE_AA)

        # Corner dots
        for pt in [sL, sR, iL, iR]:
            cv2.circle(overlay, pt, 5, col, -1, cv2.LINE_AA)

        # Vertebra label
        mid_x = (sL[0] + sR[0]) // 2
        mid_y = (sL[1] + sR[1]) // 2
        cv2.putText(overlay, v['name'], (mid_x - 30, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # Compute Cobb angles between pairs
    # Each pair = (superior terminal, inferior terminal)
    if len(vertebrae_landmarks) >= 2:
        for i in range(len(vertebrae_landmarks) - 1):
            v_top = vertebrae_landmarks[i]
            v_bot = vertebrae_landmarks[i + 1]

            angle = cobb_angle(
                v_top['sup_L'], v_top['sup_R'],
                v_bot['inf_L'], v_bot['inf_R']
            )
            results.append({
                'from': v_top['name'],
                'to': v_bot['name'],
                'angle': round(angle, 1)
            })

            # Extend endplate lines to show intersection
            # Draw perpendiculars for visual clarity
            mid_top = ((v_top['sup_L'][0]+v_top['sup_R'][0])//2,
                       (v_top['sup_L'][1]+v_top['sup_R'][1])//2)
            mid_bot = ((v_bot['inf_L'][0]+v_bot['inf_R'][0])//2,
                       (v_bot['inf_L'][1]+v_bot['inf_R'][1])//2)

            cv2.putText(overlay, f"Cobb {angle:.1f}°",
                        (max(mid_top[0], mid_bot[0]) + 10,
                         (mid_top[1] + mid_bot[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,215,0), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    return results, img


# ─────────────────────────────────────────────
#  INTERACTIVE LANDMARK EDITOR (canvas-based)
# ─────────────────────────────────────────────

def image_to_base64(img_array):
    """Convert numpy RGB array to base64 PNG string."""
    pil_img = Image.fromarray(img_array)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def landmark_editor_hka(img_array):
    """
    Interactive HTML canvas landmark editor for HKA measurement.
    Returns landmark dict or None if not confirmed.
    """
    h, w = img_array.shape[:2]
    # Scale for display
    MAX_W = 700
    scale = min(MAX_W / w, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    img_b64 = image_to_base64(cv2.resize(img_array, (dw, dh)))

    landmark_defs = [
        ("A", "Centro cabeza femoral", "#ff6464"),
        ("B", "Centro escotadura intercondílea", "#66fcf1"),
        ("C", "Centro espinas tibiales", "#66fcf1"),
        ("D", "Centro articulación tibio-astragalina", "#ff6464"),
        ("E", "Cóndilo medial femoral (distal)", "#ffa500"),
        ("F", "Cóndilo lateral femoral (distal)", "#ffa500"),
        ("G", "Platillo tibial medial", "#ffa500"),
        ("H", "Platillo tibial lateral", "#ffa500"),
    ]

    # Build JS landmark list
    lm_js = json.dumps([{"id": l[0], "label": l[1], "color": l[2]} for l in landmark_defs])

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; background:#0b0c10; font-family: sans-serif; color:#c5c6c7; }}
  #container {{ position:relative; display:inline-block; }}
  canvas {{ cursor:crosshair; border:1px solid #45a29e; border-radius:8px; }}
  #controls {{ padding:10px; }}
  .lm-btn {{
    display:inline-block; margin:4px; padding:6px 12px;
    background:rgba(31,40,51,0.8); border:1px solid #45a29e;
    border-radius:6px; color:#66fcf1; cursor:pointer; font-size:12px;
  }}
  .lm-btn.active {{ background:rgba(102,252,241,0.2); border-color:#66fcf1; }}
  .lm-btn.done {{ border-color:#2ecc71; color:#2ecc71; }}
  #status {{ padding:8px 10px; font-size:13px; color:#45a29e; }}
  #output {{ display:none; }}
</style>
</head>
<body>
<div id="controls">
  <div id="status">Selecciona un punto y haz clic en la imagen</div>
  <div id="buttons"></div>
</div>
<div id="container">
  <canvas id="cvs" width="{dw}" height="{dh}"></canvas>
</div>
<div id="output"></div>

<script>
const LANDMARKS = {lm_js};
const SCALE = {scale};
let currentLM = null;
let placed = {{}};

const cvs = document.getElementById('cvs');
const ctx = cvs.getContext('2d');
const img = new Image();
img.src = 'data:image/png;base64,{img_b64}';
img.onload = () => {{ redraw(); }};

// Build buttons
const btns = document.getElementById('buttons');
LANDMARKS.forEach(lm => {{
  const b = document.createElement('span');
  b.className = 'lm-btn';
  b.id = 'btn_' + lm.id;
  b.textContent = lm.id + ': ' + lm.label;
  b.onclick = () => {{ selectLM(lm.id); }};
  btns.appendChild(b);
}});

function selectLM(id) {{
  currentLM = id;
  document.querySelectorAll('.lm-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn_' + id);
  if (btn) btn.classList.add('active');
  document.getElementById('status').textContent =
    'Haz clic en la imagen para colocar: ' + id + ' – ' +
    LANDMARKS.find(l=>l.id===id).label;
}}

cvs.addEventListener('click', (e) => {{
  if (!currentLM) return;
  const rect = cvs.getBoundingClientRect();
  const cx = Math.round((e.clientX - rect.left));
  const cy = Math.round((e.clientY - rect.top));
  // Store original-image coordinates
  placed[currentLM] = [Math.round(cx / SCALE), Math.round(cy / SCALE)];

  const btn = document.getElementById('btn_' + currentLM);
  if (btn) {{ btn.classList.remove('active'); btn.classList.add('done'); }}
  currentLM = null;
  document.getElementById('status').textContent =
    'Punto colocado. Selecciona el siguiente.';
  redraw();
  outputCoords();
}});

function redraw() {{
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(img, 0, 0, cvs.width, cvs.height);

  // Draw axes if we have enough points
  const sc = SCALE;
  if (placed.A && placed.B) drawLine(placed.A, placed.B, '#66fcf1', 2.5);
  if (placed.C && placed.D) drawLine(placed.C, placed.D, '#66fcf1', 2.5);
  if (placed.A && placed.D) drawLine(placed.A, placed.D, '#888', 1.5, [8,6]);
  if (placed.E && placed.F) drawLine(placed.E, placed.F, '#ffa500', 2);
  if (placed.G && placed.H) drawLine(placed.G, placed.H, '#ffa500', 2);

  // Draw placed landmarks
  LANDMARKS.forEach(lm => {{
    if (placed[lm.id]) {{
      const [ox,oy] = placed[lm.id];
      const cx = ox*sc, cy = oy*sc;
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI*2);
      ctx.fillStyle = lm.color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(lm.id, cx+10, cy+5);
    }}
  }});
}}

function drawLine(p1, p2, color, width, dash=[]) {{
  const sc = SCALE;
  ctx.beginPath();
  ctx.setLineDash(dash);
  ctx.moveTo(p1[0]*sc, p1[1]*sc);
  ctx.lineTo(p2[0]*sc, p2[1]*sc);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.setLineDash([]);
}}

function outputCoords() {{
  const el = document.getElementById('output');
  el.textContent = JSON.stringify(placed);
  // Send to Streamlit via query param update trick
  window.parent.postMessage({{type:'streamlit:setComponentValue', value: JSON.stringify(placed)}}, '*');
}}
</script>
</body>
</html>
"""
    return html_content


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

def main():
    # Header
    col_logo, col_title = st.columns([1, 5])
    with col_title:
        st.markdown("""
        <h1 style="color:#66fcf1; margin:0; font-size:2.2rem;">beam AI</h1>
        <p style="color:#45a29e; margin:0; font-size:1rem; letter-spacing:2px;">
        PLATAFORMA DE MEDICIONES AUTOMATIZADAS MSK</p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Session state for landmarks
    if 'hka_landmarks' not in st.session_state:
        st.session_state.hka_landmarks = {}
    if 'cobb_vertebrae' not in st.session_state:
        st.session_state.cobb_vertebrae = []
    if 'img_array' not in st.session_state:
        st.session_state.img_array = None
    if 'pixel_spacing' not in st.session_state:
        st.session_state.pixel_spacing = None
    if 'active_mode' not in st.session_state:
        st.session_state.active_mode = None

    # ── File upload ──────────────────────────
    uploaded = st.file_uploader(
        "Sube una radiografía (DICOM, JPG, PNG)",
        type=['dcm', 'jpg', 'jpeg', 'png']
    )

    if uploaded:
        img_array, pixel_spacing = load_image(uploaded)
        st.session_state.img_array = img_array
        st.session_state.pixel_spacing = pixel_spacing

    if st.session_state.img_array is None:
        st.info("📂 Sube una radiografía para comenzar el análisis.")
        return

    img_array = st.session_state.img_array
    pixel_spacing = st.session_state.pixel_spacing

    # ── Layout: image left, controls right ───
    col_img, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.markdown('<div class="section-header">Panel Clínico IA</div>', unsafe_allow_html=True)

        if pixel_spacing:
            st.markdown(f"""<div class="landmark-info">
            📐 Pixel spacing DICOM: <strong>{pixel_spacing:.3f} mm/px</strong><br>
            Las medidas lineales serán en mm reales.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="landmark-info" style="border-color:#f39c12">
            ⚠️ Sin pixel spacing – imagen JPG/PNG.<br>
            Los ángulos son precisos; distancias en píxeles.
            </div>""", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🦴 Eje Mecánico (HKA)", "📏 Ángulo de Cobb"])

        # ────────────────────────────────────────
        # TAB 1 – HKA
        # ────────────────────────────────────────
        with tab1:
            st.markdown("""
            **Metodología (León Muñoz / Paley)**

            Marca los 8 puntos anatómicos siguiendo la guía:
            """)
            
            landmark_labels = {
                'A': ('Centro cabeza femoral', '🔴'),
                'B': ('Centro escotadura intercondílea', '🔵'),
                'C': ('Centro espinas tibiales', '🔵'),
                'D': ('Centro articulación tibio-astragalina', '🔴'),
                'E': ('Cóndilo medial femoral distal', '🟠'),
                'F': ('Cóndilo lateral femoral distal', '🟠'),
                'G': ('Platillo tibial medial', '🟠'),
                'H': ('Platillo tibial lateral', '🟠'),
            }

            st.markdown('<div class="section-header">Landmarks colocados</div>',
                        unsafe_allow_html=True)

            lm = st.session_state.hka_landmarks
            for key, (desc, icon) in landmark_labels.items():
                placed = key in lm
                status = "✅" if placed else "⭕"
                val = f" → ({lm[key][0]}, {lm[key][1]})" if placed else ""
                st.markdown(f"<small>{status} <b>{key}</b>: {desc}{val}</small>",
                            unsafe_allow_html=True)

            st.markdown("")

            # Manual coordinate entry (for demo / precise placement)
            with st.expander("📍 Ingresar coordenadas manualmente"):
                st.caption("Ingresa las coordenadas X,Y de cada punto (en píxeles de la imagen original)")
                h_img, w_img = img_array.shape[:2]
                st.caption(f"Tamaño imagen: {w_img} × {h_img} px")

                col_a, col_b = st.columns(2)
                new_lm = {}
                for i, (key, (desc, _)) in enumerate(landmark_labels.items()):
                    with col_a if i % 2 == 0 else col_b:
                        default_x = lm.get(key, [w_img//2, h_img//2])[0]
                        default_y = lm.get(key, [w_img//2, h_img//2])[1]
                        x = st.number_input(f"{key} X ({desc[:20]}…)",
                                            0, w_img, default_x, key=f"x_{key}")
                        y = st.number_input(f"{key} Y",
                                            0, h_img, default_y, key=f"y_{key}")
                        new_lm[key] = (int(x), int(y))

                if st.button("💾 Guardar landmarks HKA"):
                    st.session_state.hka_landmarks = new_lm
                    st.success("Landmarks guardados.")

            # Quick preset for demo
            if st.button("🎯 Demo – Landmarks automáticos (estimados)"):
                h_img, w_img = img_array.shape[:2]
                # Estimate landmarks based on image proportions
                # These are typical positions for a full-leg AP radiograph
                cx = w_img // 2
                st.session_state.hka_landmarks = {
                    'A': (int(cx * 0.97), int(h_img * 0.08)),   # hip center
                    'B': (int(cx * 0.99), int(h_img * 0.47)),   # intercondylar notch
                    'C': (int(cx * 1.00), int(h_img * 0.52)),   # tibial spines
                    'D': (int(cx * 1.02), int(h_img * 0.94)),   # ankle
                    'E': (int(cx * 0.93), int(h_img * 0.50)),   # medial condyle
                    'F': (int(cx * 1.06), int(h_img * 0.50)),   # lateral condyle
                    'G': (int(cx * 0.93), int(h_img * 0.53)),   # medial tibial plateau
                    'H': (int(cx * 1.06), int(h_img * 0.53)),   # lateral tibial plateau
                }
                st.success("Landmarks estimados colocados – ajusta manualmente para mayor precisión.")

            st.markdown("")
            if len(st.session_state.hka_landmarks) >= 8:
                if st.button("📐 CALCULAR Eje Mecánico HKA", type="primary"):
                    st.session_state.active_mode = 'hka'

        # ────────────────────────────────────────
        # TAB 2 – COBB
        # ────────────────────────────────────────
        with tab2:
            st.markdown("""
            **Metodología (Informe Estructurado / Lenke)**

            Marca los platillos de las vértebras terminales de cada curva.
            """)

            st.markdown('<div class="section-header">Vértebras registradas</div>',
                        unsafe_allow_html=True)

            for i, v in enumerate(st.session_state.cobb_vertebrae):
                st.markdown(f"<small>✅ <b>{v['name']}</b> – platillos definidos</small>",
                            unsafe_allow_html=True)

            with st.expander("➕ Agregar vértebra terminal"):
                vname = st.text_input("Nombre vértebra (ej: T5, T11, L4)", key="vname")
                st.caption("Platillo superior:")
                col1, col2 = st.columns(2)
                h_img, w_img = img_array.shape[:2]
                with col1:
                    sLx = st.number_input("Sup. L – X", 0, w_img, w_img//3, key="sLx")
                    sLy = st.number_input("Sup. L – Y", 0, h_img, h_img//3, key="sLy")
                with col2:
                    sRx = st.number_input("Sup. R – X", 0, w_img, w_img*2//3, key="sRx")
                    sRy = st.number_input("Sup. R – Y", 0, h_img, h_img//3, key="sRy")

                st.caption("Platillo inferior:")
                col3, col4 = st.columns(2)
                with col3:
                    iLx = st.number_input("Inf. L – X", 0, w_img, w_img//3, key="iLx")
                    iLy = st.number_input("Inf. L – Y", 0, h_img, h_img//3+30, key="iLy")
                with col4:
                    iRx = st.number_input("Inf. R – X", 0, w_img, w_img*2//3, key="iRx")
                    iRy = st.number_input("Inf. R – Y", 0, h_img, h_img//3+30, key="iRy")

                if st.button("➕ Agregar vértebra"):
                    if vname:
                        st.session_state.cobb_vertebrae.append({
                            'name': vname,
                            'sup_L': (sLx, sLy),
                            'sup_R': (sRx, sRy),
                            'inf_L': (iLx, iLy),
                            'inf_R': (iRx, iRy),
                        })
                        st.success(f"Vértebra {vname} agregada.")

            if st.button("🗑️ Limpiar vértebras"):
                st.session_state.cobb_vertebrae = []

            # Demo preset
            if st.button("🎯 Demo – Escoliosis toraco-lumbar"):
                h_img, w_img = img_array.shape[:2]
                cx = w_img // 2
                # Simulate a right thoracic curve T5-T12 and lumbar curve T12-L4
                st.session_state.cobb_vertebrae = [
                    {
                        'name': 'T5 (terminal sup.)',
                        'sup_L': (int(cx*0.82), int(h_img*0.22)),
                        'sup_R': (int(cx*1.12), int(h_img*0.20)),
                        'inf_L': (int(cx*0.83), int(h_img*0.24)),
                        'inf_R': (int(cx*1.13), int(h_img*0.22)),
                    },
                    {
                        'name': 'T11 (ápex/terminal inf.)',
                        'sup_L': (int(cx*0.90), int(h_img*0.46)),
                        'sup_R': (int(cx*1.08), int(h_img*0.43)),
                        'inf_L': (int(cx*0.91), int(h_img*0.48)),
                        'inf_R': (int(cx*1.09), int(h_img*0.45)),
                    },
                    {
                        'name': 'L4 (terminal inf.)',
                        'sup_L': (int(cx*0.95), int(h_img*0.72)),
                        'sup_R': (int(cx*1.05), int(h_img*0.70)),
                        'inf_L': (int(cx*0.94), int(h_img*0.74)),
                        'inf_R': (int(cx*1.06), int(h_img*0.72)),
                    },
                ]
                st.success("Vértebras demo colocadas.")

            if len(st.session_state.cobb_vertebrae) >= 2:
                if st.button("📏 CALCULAR Ángulo de Cobb", type="primary"):
                    st.session_state.active_mode = 'cobb'

    # ── Image display & results ───────────────
    with col_img:
        if st.session_state.active_mode == 'hka' and len(st.session_state.hka_landmarks) >= 8:
            with st.spinner("Calculando ejes mecánicos..."):
                try:
                    results = compute_hka(img_array, st.session_state.hka_landmarks)
                    st.image(results['image'], caption="Análisis Eje Mecánico HKA",
                             use_container_width=True)

                    # Results panel
                    st.markdown('<div class="section-header">Resultados – Plano Coronal</div>',
                                unsafe_allow_html=True)

                    hka = results['HKA']
                    interp, status = hka_interpretation(hka)
                    metric_card("HKA – Ángulo Fémoro-Tibial Mecánico",
                                f"{hka}", "°",
                                f"Normal: 177–183° | {interp}",
                                status)

                    alfa = results['alfa']
                    ai, as_ = alpha_interpretation(alfa)
                    metric_card("Ángulo Alfa (Distal Femoral Lateral – medial)",
                                f"{alfa}", "°",
                                f"Normal: 84 ± 3° | {ai}", as_)

                    beta = results['beta']
                    bi, bs = beta_interpretation(beta)
                    metric_card("Ángulo Beta (Proximal Tibial Mecánico)",
                                f"{beta}", "°",
                                f"Normal: 87 ± 3° | {bi}", bs)

                    mad_px = results['MAD_px']
                    mad_side = results['MAD_side']
                    if pixel_spacing:
                        mad_mm = round(mad_px * pixel_spacing, 1)
                        metric_card("MAD – Desviación Eje Mecánico en Rodilla",
                                    f"{mad_mm}", "mm",
                                    f"Dirección: {mad_side} | Normal: <10 mm",
                                    "ok" if mad_mm < 10 else "warn")
                    else:
                        metric_card("MAD – Desviación Eje Mecánico",
                                    f"{mad_px}", "px",
                                    f"Dirección: {mad_side} (sin pixel spacing)")

                    # Knee Society note
                    st.info(f"📋 **Knee Society**: sin penalización entre 178° (2° varo) y 190° (10° valgo). "
                            f"HKA actual: **{hka}°**")

                    # Export
                    report = {
                        "estudio": "Eje Mecánico Extremidad Inferior",
                        "metodologia": "León Muñoz / Paley / Knee Society",
                        "HKA_grados": hka,
                        "interpretacion_HKA": interp,
                        "angulo_alfa_grados": alfa,
                        "angulo_beta_grados": beta,
                        "MAD_px": mad_px,
                        "MAD_side": mad_side,
                        "pixel_spacing_mm": pixel_spacing,
                    }
                    st.download_button(
                        "⬇️ Descargar reporte JSON",
                        data=json.dumps(report, ensure_ascii=False, indent=2),
                        file_name="beam_ai_hka_report.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error en cálculo: {e}")
                    st.info("Verifica que todos los landmarks estén bien colocados.")

        elif st.session_state.active_mode == 'cobb' and len(st.session_state.cobb_vertebrae) >= 2:
            with st.spinner("Calculando ángulo de Cobb..."):
                try:
                    cobb_results, cobb_img = compute_cobb(
                        img_array, st.session_state.cobb_vertebrae)
                    st.image(cobb_img, caption="Análisis Ángulo de Cobb",
                             use_container_width=True)

                    st.markdown('<div class="section-header">Resultados – Ángulo de Cobb</div>',
                                unsafe_allow_html=True)

                    for r in cobb_results:
                        interp, status = cobb_interpretation(r['angle'])
                        metric_card(
                            f"Cobb: {r['from']} → {r['to']}",
                            f"{r['angle']}", "°",
                            f"{interp}", status
                        )

                    # Lenke classification hint
                    st.markdown('<div class="section-header">Orientación Clasificación Lenke</div>',
                                unsafe_allow_html=True)

                    if cobb_results:
                        max_cobb = max(r['angle'] for r in cobb_results)
                        st.markdown(f"""
                        <div class="landmark-info">
                        <b>Curva mayor:</b> {max(cobb_results, key=lambda x: x['angle'])['from']} →
                        {max(cobb_results, key=lambda x: x['angle'])['to']} ({max_cobb}°)<br>
                        <b>Curvas menores estructurales si:</b> Cobb ≥ 25° o cifosis ≥ 20°<br>
                        <b>Progresión:</b> Variación >6° respecto a estudio previo<br>
                        <small>Seguimiento adultos con curva >30°: cada 5 años (Estrada et al. 2017)</small>
                        </div>
                        """, unsafe_allow_html=True)

                    report_cobb = {
                        "estudio": "Escoliosis – Ángulo de Cobb",
                        "metodologia": "Informe Estructurado / Clasificación Lenke",
                        "curvas": cobb_results,
                        "vertebras_evaluadas": [v['name'] for v in st.session_state.cobb_vertebrae],
                    }
                    st.download_button(
                        "⬇️ Descargar reporte JSON",
                        data=json.dumps(report_cobb, ensure_ascii=False, indent=2),
                        file_name="beam_ai_cobb_report.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error en cálculo: {e}")

        else:
            # Show original image with guide overlay
            guide_img = to_rgb(img_array).copy()
            h_img, w_img = guide_img.shape[:2]

            # Watermark
            cv2.putText(guide_img, "beam AI – MSK Platform",
                        (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (69, 162, 158), 2, cv2.LINE_AA)

            st.image(guide_img, caption="Imagen cargada – selecciona análisis en el panel derecho",
                     use_container_width=True)

            if pixel_spacing:
                st.success(f"✅ DICOM con pixel spacing: {pixel_spacing:.3f} mm/px — medidas en mm reales")

    # ── Footer ─────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#45a29e; font-size:12px;">
    <b>beam AI</b> · Mediciones MSK automatizadas ·
    Metodología: León Muñoz (SEROD 2017) · Estrada et al. (HPTU 2017) ·
    Paley · Knee Society · Clasificación de Lenke<br>
    <span style="color:#555">⚠️ Solo para uso investigativo y educativo.
    Las mediciones clínicas deben ser validadas por un especialista.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

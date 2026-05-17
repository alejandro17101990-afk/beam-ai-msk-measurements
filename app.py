import numpy as np

def calcular_hka(cadera, rodilla, tobillo):
    """
    cadera, rodilla, tobillo: tuplas (x, y) en píxeles
    Retorna ángulo HKA en grados
    """
    v1 = np.array(cadera) - np.array(rodilla)   # vector femoral
    v2 = np.array(tobillo) - np.array(rodilla)  # vector tibial
    
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angulo = np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))
    
    # HKA normal ≈ 180° (neutro), varo > 180°, valgo < 180°
    return angulo

def calcular_cobb(punto_superior_izq, punto_superior_der, punto_inferior_izq, punto_inferior_der):
    """
    Retorna el ángulo de Cobb entre dos vértebras en grados
    """
    angulo_sup = np.degrees(np.arctan2(
        punto_superior_der[1] - punto_superior_izq[1],
        punto_superior_der[0] - punto_superior_izq[0]
    ))
    angulo_inf = np.degrees(np.arctan2(
        punto_inferior_der[1] - punto_inferior_izq[1],
        punto_inferior_der[0] - punto_inferior_izq[0]
    ))
    return abs(angulo_sup - angulo_inf)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box # creeaza un poligon cu colturile date
from shapely.ops import unary_union # face uniunea geometrica, elimina zonele comune

# Dreptunghiul real (ground truth) – desenat in albastru
real_box_coords = (1, 1, 11, 11)  # format: (x1, y1, x2, y2)
real_poly = box(*real_box_coords) # din tuplu il face despachetat

# Lista cu coordonatele detectiilor (pot fi oricate)
det_coords_list = [
    (3, 3, 7, 7),    # Detectie 1
    (6, 4, 9, 8),    # Detectie 2 (se suprapune partial cu Detectie 1)
    (2, 8, 4, 12)    # Detectie 3 (acum iese putin din dreptunghiul real)
]

# Convertim fiecare detectie intr-un poligon shapely
det_polys = [box(*coords) for coords in det_coords_list]

# Calculam uniunea geometrica a tuturor detectiilor -> multipolygon pt ca este Detectie 3 care nu se suprapune
detections_union = unary_union(det_polys)

# Calculam intersectia dintre dreptunghiul real si uniunea detectiilor
intersection = real_poly.intersection(detections_union)

# Calculam ariile necesare pentru IoU
A_real = real_poly.area
A_det = detections_union.area
I = intersection.area
Union_total = A_real + A_det - I
iou = I / Union_total if Union_total > 0 else 0.0

# Cream graficul
fig, ax = plt.subplots(figsize=(8, 8))

# Desenam dreptunghiul real (albastru)
real_rect = patches.Rectangle((real_box_coords[0], real_box_coords[1]),
                              real_box_coords[2] - real_box_coords[0],
                              real_box_coords[3] - real_box_coords[1],
                              edgecolor='blue', facecolor='none', linewidth=2, label='Real')
ax.add_patch(real_rect)

# Desenam fiecare detectie individual (rosu)
for i, coords in enumerate(det_coords_list):
    rect = patches.Rectangle((coords[0], coords[1]),
                             coords[2] - coords[0],
                             coords[3] - coords[1],
                             edgecolor='red', facecolor='none', linewidth=2,
                             label='Detectie' if i == 0 else None)
    ax.add_patch(rect)

# Desenam conturul uniunii detectiilor (verde, linie întrerupta)
if detections_union.geom_type == 'Polygon':
    x, y = detections_union.exterior.xy
    ax.plot(x, y, color='green', linestyle='--', linewidth=2, label='Uniunea detectiilor')
elif detections_union.geom_type == 'MultiPolygon':
    for poly in detections_union.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='green', linestyle='--', linewidth=2, label='Uniunea detectiilor')

# Desenam intersectia (zona comuna intre real si uniunea detectiilor) – umpluta cu mov transparent
if not intersection.is_empty:
    if intersection.geom_type == 'Polygon':
        x, y = intersection.exterior.xy
        ax.fill(x, y, color='purple', alpha=0.3, label='Intersectie')
    elif intersection.geom_type == 'MultiPolygon':
        for poly in intersection.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color='purple', alpha=0.3, label='Intersectie')

# Adăugam pe grafic valorile ariilor si IoU
textstr = (f"Aria Real: {A_real:.2f}\n"
           f"Aria Detectiilor: {A_det:.2f}\n"
           f"Aria Intersectiei: {I:.2f}\n"
           f"Union Total: {Union_total:.2f}\n"
           f"IoU: {iou:.2f}")

ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Configurăm axele și legenda
ax.set_xlim(0, 13)
ax.set_ylim(0, 13)
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.set_title("Ilustrarea IoU: Real vs. Uniunea Detectiilor")
plt.show()

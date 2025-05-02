import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmocr.apis import MMOCRInferencer
from skimage import io
from shapely.geometry import box
from shapely.ops import unary_union
import numpy as np
import pprint

# Folderul cu imaginile
input_folder = r"C:\Users\denis\Desktop\sample_images\ROC_batch_2a\210308A0E5DB"
output_folder = r"C:\Users\denis\Desktop\sample_images\saves"
os.makedirs(output_folder, exist_ok=True)


# Cele 3 dreptunghiuri reale (ground truth) – fiecare definit ca un poligon cu 8 coordonate
real_polygons = [
    [1092, 1104, 1092, 1215, 1653, 1215, 1653, 1104],
    [1050, 1257, 1050, 1380, 1698, 1380, 1698, 1257],
    [1191, 1425, 1191, 1581, 1569, 1581, 1569, 1425]
]

# Initializare detector
detector = "textsnake"
infer = MMOCRInferencer(det=detector, rec=None, device='cpu')

# Functie: transforma un poligon (lista de coordonate) intr-un bounding box [x1, y1, x2, y2]
def corners_to_bbox(polygon):
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

# Obtine prima imagine din folder
image_extensions = (".png", ".jpg", ".jpeg")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

if image_files:
    first_image = image_files[0]
    file_path = os.path.join(input_folder, first_image)
    image = io.imread(file_path)
    

    result = infer(file_path)
    pprint.pprint(result)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    # Convertim detectiile la bounding boxes (lista de [x1, y1, x2, y2])
    detected_boxes = []
    if result and 'predictions' in result and result['predictions'][0]['det_polygons']:
        polygons = result['predictions'][0]['det_polygons']
        scores = result['predictions'][0]['det_scores']

        for det_poly, score in zip(polygons, scores):
            if score >= 0.5:
                bbox = corners_to_bbox(det_poly)
                dx1, dy1, dx2, dy2 = bbox

                # Ignoram cutiile foarte aproape de coltul stanga sus
                if dy1 < 900:
                    continue  # Sare peste aceasta detectie

                detected_boxes.append(bbox)
                ax.add_patch(plt.Rectangle((dx1, dy1), dx2 - dx1, dy2 - dy1,
                                        edgecolor='red', facecolor='none', linewidth=2, label='Detectat'))


    # Lista pentru a stoca IoU-ul calculat pentru fiecare obiect real
    iou_list = []

    # Pentru fiecare dreptunghi real, calculam IoU-ul pe baza uniunii detectiilor ce se intersectează
    for polygon in real_polygons:
        # Convertim poligonul real in bounding box si apoi intr-un obiect Shapely
        real_bbox = corners_to_bbox(polygon)
        x1, y1, x2, y2 = real_bbox
        real_area = (x2 - x1) * (y2 - y1)
        real_shapely = box(x1, y1, x2, y2)
        
        # Selectam detectiile care intersecteaza obiectul real
        detection_polys = []
        for det_box in detected_boxes:
            det_shapely = box(*det_box) # box(det_box[0], det_box[1], det_box[2], det_box[3])
            if real_shapely.intersects(det_shapely): # daca fiecare detectie mica rosie se intersecteaza cu cel mare albastru
                detection_polys.append(det_shapely)
        
        if detection_polys:
            # Calculam uniunea geometrică a detectiilor ce se intersecteaza cu cel albastru mare
            # daca avem 2 rosii mici care se intersecteaza, se calculeaza uniunea lor = arie1+arie2-inter
            # daca e separat vreunul, doar aduna cu aria lui si atat
            # deci va fi uniunea pt toate detectiile rosii mici fara dubluri
            detections_union = unary_union(detection_polys)
            # Calculăm intersecția dintre real și uniunea detecțiilor
            intersection = real_shapely.intersection(detections_union)
            A_det = detections_union.area
            I = intersection.area
            union_total = real_area + A_det - I
            iou = I / union_total if union_total > 0 else 0.0
            iou_list.append(iou)
            
            # Desenam conturul uniunii detectiilor (verde, linie întrerupta)
            if detections_union.geom_type == 'Polygon':
                x, y = detections_union.exterior.xy
                ax.plot(x, y, color='green', linestyle='--', linewidth=2, label='Uniunea Detectiilor')
            elif detections_union.geom_type == 'MultiPolygon':
                for poly in detections_union.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='green', linestyle='--', linewidth=2, label='Uniunea Detectiilor')
            
            # Desenam intersectia (umpluta cu mov transparent)
            if not intersection.is_empty:
                if intersection.geom_type == 'Polygon':
                    x, y = intersection.exterior.xy
                    ax.fill(x, y, color='purple', alpha=0.3, label='Intersectie')
                elif intersection.geom_type == 'MultiPolygon':
                    for poly in intersection.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='purple', alpha=0.3, label='Intersectie')


            # Luam bounding box-ul detections_union
            minx, miny, maxx, maxy = detections_union.bounds

            # Asiguram ca sunt intregi si in limitele imaginii
            minx = max(0, int(minx))
            miny = max(0, int(miny))
            maxx = min(image.shape[1], int(maxx))
            maxy = min(image.shape[0], int(maxy))

            # Crop direct din imagine
            cropped_image = image[miny:maxy, minx:maxx]

            # Salvam imaginea croppuita
            save_path = os.path.join(output_folder, f"detections_union_crop_real_{real_polygons.index(polygon)+1}.png")
            io.imsave(save_path, cropped_image)            
        else:
            iou_list.append(0.0)
        
        # Desenam dreptunghiul real (albastru)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='blue', facecolor='none', linewidth=2, label='Real'))
    
    # Calculam media IoU-urilor (opțional)
    avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    
    # Afisam pe grafic IoU pentru fiecare obiect si media
    textstr = f"IoU per obiect: {', '.join([f'{val:.2f}' for val in iou_list])}\nMedia IoU: {avg_iou:.2f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legendă unică
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    
    plt.title(f"{first_image} - IoU pe obiecte")
    plt.show()
else:
    print("Nu s-au gasit imagini in folder.")

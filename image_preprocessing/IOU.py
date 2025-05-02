import os
from mmocr.apis import MMOCRInferencer
from skimage import io
from shapely.geometry import box
from shapely.ops import unary_union

# Folderul cu imaginile si fisierul de iesire (se suprascrie de fiecare data)
input_folder = r"C:\Users\denis\Desktop\sample_images\ROC_batch_2a\210308A0E5DB"
output_file = r"C:\Users\denis\Desktop\iou_results.txt"

# Cele 3 dreptunghiuri reale (ground truth) – fiecare definit ca un poligon cu 8 coordonate
real_polygons = [
    [1092, 1104, 1092, 1215, 1653, 1215, 1653, 1104],
    [1050, 1257, 1050, 1380, 1698, 1380, 1698, 1257],
    [1191, 1425, 1191, 1581, 1569, 1581, 1569, 1425]
]

# Initializare detector
detector = "dbnetpp"
infer = MMOCRInferencer(det=detector, rec=None, device='cpu')

# Functie: transforma un poligon (lista de coordonate) intr-un bounding box [x1, y1, x2, y2]
def corners_to_bbox(polygon):
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

# Filtram fisierele de imagine din folder
image_extensions = (".png", ".jpg", ".jpeg")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

# Lista pentru a stoca rezultatele sub forma unui tuplu: (numele imaginii, IoU mediu)
results = []

for image_file in image_files:
    file_path = os.path.join(input_folder, image_file)
    
    # Efectuam inferenta pe imagine
    result = infer(file_path)
    
    # Extragem detectiile sub forma de bounding boxes [x1, y1, x2, y2]
    detected_boxes = []
    if result and 'predictions' in result and result['predictions'][0]['det_polygons']:
        for det_poly in result['predictions'][0]['det_polygons']:
            bbox = corners_to_bbox(det_poly)
            detected_boxes.append(bbox)
    
    # Calculam IoU pentru fiecare obiect real
    iou_list = []
    for polygon in real_polygons:
        # Convertim poligonul real în bounding box si obiect Shapely
        real_bbox = corners_to_bbox(polygon)
        x1, y1, x2, y2 = real_bbox
        real_area = (x2 - x1) * (y2 - y1)
        real_shapely = box(x1, y1, x2, y2)
        
        # Selectam detectiile care intersecteaza cu obiectul real
        detection_polys = []
        for det_box in detected_boxes:
            det_shapely = box(*det_box)
            if real_shapely.intersects(det_shapely):
                detection_polys.append(det_shapely)
        
        if detection_polys:
            # Uniunea geometrica a detectiilor intersectate
            detections_union = unary_union(detection_polys)
            # Intersectia dintre obiectul real si uniunea detectiilor
            intersection = real_shapely.intersection(detections_union)
            A_det = detections_union.area
            I = intersection.area
            union_total = real_area + A_det - I
            iou = I / union_total if union_total > 0 else 0.0
            iou_list.append(iou)
        else:
            iou_list.append(0.0)
    
    # Media IoU-urilor pentru imagine (fiecare obiect real are un IoU, se calculează media lor)
    avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    results.append((image_file, avg_iou))

# Scriem rezultatele intr-un fisier (suprascriere)
with open(output_file, 'w') as f:
    for image_file, avg_iou in results:
        f.write(f"{image_file}, {avg_iou:.2f}\n")

print("Procesare finalizata. Rezultatele IoU au fost salvate in:", output_file)

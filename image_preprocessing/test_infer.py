import os
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer
from skimage import io

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\sample_images\crop"
output_base = r"C:\Users\denis\Desktop\sample_images\detectie\det\fara_filtre"

# Lista detectorilor disponibili
detectors = ["dbnet_resnet18_fpnc_100k_synthtext"]

# Tipuri de fisiere imagine acceptate
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Parcurge toate subfolderele relevante
for root, _, files in os.walk(input_folder):
    # Gaseste prima imagine valida
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    if image_files:  # Verifica daca exista cel putin o imagine in folder
        first_image = image_files[0]  # Prima imagine
        file_path = os.path.join(root, first_image)

        # Încarcă imaginea și verifică dacă este validă
        image = io.imread(file_path)

        if image is None or image.size == 0:
            print(f"Eroare: Imaginea {file_path} este goală sau nu a fost încărcată corect! Se sare peste aceasta.")
            continue

        # Verifică dimensiunile imaginii
        if len(image.shape) < 2:
            print(f"Eroare: Imaginea {file_path} are dimensiuni invalide! Se sare peste aceasta.")
            continue

        # Obtine structura relativa a folderului
        relative_path = os.path.relpath(root, input_folder)

        # Ruleaza inferenta pentru fiecare detector
        for detector in detectors:
            print(f"Procesare cu {detector} pentru {first_image}...")

            # Creeaza folderul de output pentru detector
            output_folder = os.path.join(output_base, detector, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            # Creeaza obiectul de inferenta
            infer = MMOCRInferencer(det=detector, rec='svtr-small', device='cpu')

            # Ruleaza inferenta pe imagine
            try:
                result = infer(file_path, save_vis=True, return_vis=True)

                # Verificare: Dacă inferența nu returnează rezultate valide, se sare peste imagine
                if not result or 'visualization' not in result or not result['visualization']:
                    print(f"Eroare: Inferența nu a returnat rezultate pentru {file_path}. Se sare peste aceasta.")
                    continue

                # Obtine imaginea rezultata
                detected_image = result['visualization'][0]

                # Salveaza imaginea finala cu acelasi nume
                output_path = os.path.join(output_folder, first_image)
                plt.imsave(output_path, detected_image)

                print(f"Salvat: {output_path}")

            except Exception as e:
                print(f"Eroare la procesarea {file_path} cu {detector}: {e}")
                continue  # Continuă cu următoarea imagine

print("Toate imaginile au fost procesate!")

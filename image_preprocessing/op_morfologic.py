import os
import numpy as np
from skimage import io, morphology

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\sample_images\otsu"
output_folder = r"C:\Users\denis\Desktop\sample_images\op_morfologic"

# Creează folderul de ieșire dacă nu există
os.makedirs(output_folder, exist_ok=True)

# Tipuri de fișiere imagine acceptate
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Parcurgem toate subfolderele și procesăm toate imaginile
for root, _, files in os.walk(input_folder):
    # Filtrăm doar imaginile
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    if image_files:
        # Creăm structura de foldere pentru output
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for filename in image_files:
            file_path = os.path.join(root, filename)

            # Încarcă imaginea binară (pragul Otsu aplicat anterior)
            image = io.imread(file_path, as_gray=True)
            binary_image = image // 255  # Convertim în imagine binară

            # Aplicăm închiderea morfologică pentru a elimina găurile mici
            selem = morphology.disk(2)  # Element structural (poți încerca valori mai mari)
            closed_image = morphology.binary_closing(binary_image, selem)

            # Aplicăm umplerea găurilor pentru a umple zonele goale din interiorul obiectelor
            filled_image = morphology.remove_small_holes(closed_image, area_threshold=100)

            # Salvăm imaginea procesată în folderul corespunzător
            output_path = os.path.join(output_subfolder, filename)
            io.imsave(output_path, filled_image.astype(np.uint8) * 255)

            print(f"Procesat: {file_path} -> Salvata în {output_subfolder}")

print("Toate imaginile au fost procesate!")

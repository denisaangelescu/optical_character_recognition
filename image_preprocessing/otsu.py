import os
import cv2
import numpy as np
from skimage import io

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\sample_images\filtru_median"
output_folder = r"C:\Users\denis\Desktop\sample_images\otsu"

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

            # Încarcă imaginea în grayscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Inversează imaginea (vizual)
            image = 255 - image

            # Aplică binarizarea Otsu cu prag custom
            otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            custom_thresh = min(255, max(0, otsu_thresh + 30))  # Asigură că pragul rămâne în [0, 255]

            _, thresholded = cv2.threshold(image, custom_thresh, 255, cv2.THRESH_BINARY)

            # Salvăm imaginea binarizată în același format, păstrând structura de foldere
            output_path = os.path.join(output_subfolder, filename)
            cv2.imwrite(output_path, thresholded)

            print(f"Procesat: {file_path} -> Salvata în {output_subfolder}")

print("Toate imaginile au fost procesate cu Otsu!")

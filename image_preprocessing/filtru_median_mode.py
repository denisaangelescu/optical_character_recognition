import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import io, color, img_as_ubyte

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\incercari_filtre\i1"
output_folder = os.path.join(input_folder, "filtru_median")

# Creează folderul de ieșire dacă nu există
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tipuri de fișiere imagine pe care le procesăm
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Funcție pentru conversia corectă la grayscale (gestionând și RGBA)
def convert_to_grayscale(image):
    if len(image.shape) == 3:  # Dacă are mai multe canale
        if image.shape[2] == 4:  # Dacă este RGBA
            image = image[:, :, :3]  # Eliminăm canalul Alpha
        image = color.rgb2gray(image)  # Convertim la grayscale
    return image

# Lista modurilor de extindere a marginilor
modes = {
    "reflect": {"mode": "reflect"},
    "constant": {"mode": "constant", "cval": 0},
    "nearest": {"mode": "nearest"},
    "mirror": {"mode": "mirror"},
    "wrap": {"mode": "wrap"}
}

# Parcurge toate fișierele din folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):  # Verifică dacă este imagine
        file_path = os.path.join(input_folder, filename)  # Calea completă a fișierului

        # Încarcă imaginea și convertește-o la grayscale
        image = io.imread(file_path)
        image = convert_to_grayscale(image)

        # Aplică filtrul median cu size=7 (default)
        filtered_image = median_filter(image, size=7)
        output_path = os.path.join(output_folder, filename.replace(".png", "_size7.png"))
        io.imsave(output_path, img_as_ubyte(filtered_image))

        # Aplică filtrul median cu diferite moduri și salvează fiecare variantă
        for mode_name, params in modes.items():
            filtered_mode = median_filter(image, size=7, **params)
            mode_output_path = os.path.join(output_folder, filename.replace(".png", f"_mode_{mode_name}.png"))
            io.imsave(mode_output_path, img_as_ubyte(filtered_mode))

        print(f"Filtru median aplicat: {filename} -> Salvate în {output_folder}")

print("Procesarea a fost finalizată!")

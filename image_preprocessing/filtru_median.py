import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import io, color, img_as_ubyte

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\sample_images\crop"
output_folder = r"C:\Users\denis\Desktop\sample_images\filtru_median"
hist_folder = r"C:\Users\denis\Desktop\sample_images\histograme\filtru_median"

# Creează folderele principale dacă nu există
os.makedirs(output_folder, exist_ok=True)
os.makedirs(hist_folder, exist_ok=True)

# Tipuri de fișiere imagine acceptate
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Funcție pentru conversia corectă la grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 3:  # Dacă are mai multe canale
        if image.shape[2] == 4:  # Dacă este RGBA
            image = image[:, :, :3]  # Eliminăm canalul Alpha
        image = color.rgb2gray(image)  # Convertim la grayscale
    return image

# Parcurge toate subfolderele și procesează toate imaginile
for root, _, files in os.walk(input_folder):
    # Filtrăm doar imaginile
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    if image_files:
        # Creăm structura de foldere pentru output și histograme
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        hist_subfolder = os.path.join(hist_folder, relative_path)

        os.makedirs(output_subfolder, exist_ok=True)
        os.makedirs(hist_subfolder, exist_ok=True)

        for filename in image_files:
            file_path = os.path.join(root, filename)

            # Încarcă imaginea și convertește-o la grayscale
            image = io.imread(file_path)
            image = convert_to_grayscale(image)

            # Aplică filtrul median cu size=3 și mode="constant", cval=0
            filtered_image = median_filter(image, size=3, mode="constant", cval=0)

            # Salvăm imaginea filtrată în același format, păstrând structura de foldere
            output_path = os.path.join(output_subfolder, filename)
            io.imsave(output_path, img_as_ubyte(filtered_image))

            # Generare histogramă comparativă
            plt.figure(figsize=(8, 6))
            plt.hist(image.ravel(), bins=256, color="red", alpha=0.6, label="Original")
            plt.hist(filtered_image.ravel(), bins=256, color="blue", alpha=0.6, label="Filtered")
            plt.xlabel("Intensitate pixel")
            plt.ylabel("Număr de pixeli")
            plt.title(f"Histogramă: {filename}")
            plt.legend()
            plt.grid(True)

            # Salvăm histograma în folderul dedicat
            hist_path = os.path.join(hist_subfolder, f"{os.path.splitext(filename)[0]}_histogram.png")
            plt.savefig(hist_path)
            plt.close()

            print(f"Procesat: {file_path} -> Salvate în {output_subfolder} și {hist_subfolder}")

print("Toate imaginile din crop au fost procesate!")

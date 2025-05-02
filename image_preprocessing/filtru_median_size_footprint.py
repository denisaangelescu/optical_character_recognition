import os
import numpy as np
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

# Definim footprint personalizat (formă de diamant)
footprint_cross = np.array([[0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0]])  # Diamant 5x5

# Parcurge toate fișierele din folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):  # Verifică dacă este imagine
        file_path = os.path.join(input_folder, filename)  # Calea completă a fișierului
        name, ext = os.path.splitext(filename)  # Separă numele și extensia fișierului
        
        # Încarcă imaginea și convertește-o la grayscale
        image = io.imread(file_path)
        image = convert_to_grayscale(image)

        # Aplică filtrul median cu diferite dimensiuni și footprint
        filtered_5 = median_filter(image, size=5)
        filtered_7 = median_filter(image, size=7)
        filtered_9 = median_filter(image, size=9)
        filtered_11 = median_filter(image, size=11)
        filtered_cross = median_filter(image, footprint=footprint_cross)

        # Salvăm imaginile filtrate cu nume specifice metodei folosite
        io.imsave(os.path.join(output_folder, f"{name}_size5{ext}"), img_as_ubyte(filtered_5))
        io.imsave(os.path.join(output_folder, f"{name}_size7{ext}"), img_as_ubyte(filtered_7))
        io.imsave(os.path.join(output_folder, f"{name}_size9{ext}"), img_as_ubyte(filtered_9))
        io.imsave(os.path.join(output_folder, f"{name}_size11{ext}"), img_as_ubyte(filtered_11))
        io.imsave(os.path.join(output_folder, f"{name}_footprint_cross{ext}"), img_as_ubyte(filtered_cross))

        print(f"Filtre aplicate: {filename} -> Salvate în {output_folder}")

print("Procesarea a fost finalizată!")

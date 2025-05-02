import os
from skimage import io, img_as_ubyte

# Definirea folderelor
input_folder = r"C:\Users\denis\Desktop\sample_images"
output_folder = os.path.join(input_folder, "crop")

# Creeaza folderul de iesire daca nu exista
os.makedirs(output_folder, exist_ok=True)

# Lista de directoare specifice pe care le procesam
valid_folders = ["ROC_batch_2a", "ROC_batch_2b"]

# Tipuri de fisiere imagine acceptate
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Parcurge doar directoarele ROC_batch_2a și ROC_batch_2b
for batch_folder in valid_folders:
    batch_path = os.path.join(input_folder, batch_folder)  # Ex: sample_images/ROC_batch_2a
    
    if not os.path.exists(batch_path):
        continue  # Sarim peste daca folderul nu exista

    for subfolder in os.listdir(batch_path):
        subfolder_path = os.path.join(batch_path, subfolder)  # Ex: sample_images/ROC_batch_2a/210308A0E5DB

        if os.path.isdir(subfolder_path):  # Asiguram ca e folder, nu fisier
            output_subfolder = os.path.join(output_folder, batch_folder, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            # Parcurgem toate imaginile din subfolder
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(image_extensions):  
                    file_path = os.path.join(subfolder_path, filename)

                    # Incarca imaginea
                    image = io.imread(file_path)

                    # Obtine dimensiunile imaginii
                    height, width = image.shape[:2]

                    # Pasul 1: Pastram partea de jos + o optime din partea de sus
                    top_cut = height // 2 - height // 8  
                    cropped_horizontal = image[top_cut:, :]

                    # Pasul 2: Eliminam 1/4 din stanga si 1/4 din dreapta
                    left_cut = width // 4
                    right_cut = width - width // 4
                    cropped_final = cropped_horizontal[:, left_cut:right_cut]

                    # Pasul 3: Transformam in patrat pe centru
                    new_height, new_width = cropped_final.shape[:2]
                    square_size = min(new_height, new_width)

                    center_x, center_y = new_width // 2, new_height // 2
                    half_size = square_size // 2

                    square_crop = cropped_final[
                        center_y - half_size:center_y + half_size,
                        center_x - half_size:center_x + half_size
                    ]

                    # Salvam imaginea finala cu acelasi nume in structura pastrata
                    output_path = os.path.join(output_subfolder, filename)
                    io.imsave(output_path, img_as_ubyte(square_crop))

                    print(f"Procesata: {file_path} -> Salvata ca {output_path}")

print("Toate imaginile din ROC_batch_2a și ROC_batch_2b au fost procesate!")

import json
import os
from mmocr.apis import TextRecInferencer

# functie pentru rularea inferentei si salvarea predictiilor
def save_predictions(config_path, checkpoint_path, image_list, output_file):
    # inifializează MMOCR cu fisierul de configurare și checkpoint-ul antrenat
    infer = TextRecInferencer(model=config_path, weights=checkpoint_path, scope='mmocr') # obiect pentru inferenta
    results = [] # lista goala

    # parcurge toate imaginile și ruleaza inferenta
    for img_path in image_list:
        result = infer(img_path)  # aplică OCR pe imagine si returneaza rezultatele
        pred_text = result['predictions'][0]['text']  # extrage textul prezis de model
        pred_text = pred_text.replace(" ", "").upper() # elimina spatiile si converteste in majuscule (ca la json)
        results.append({'image': img_path, 'predicted_text': pred_text}) # salveaza intr o lista

    # Salvează rezultatele într-un fișier JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predicțiile au fost salvate în {output_file}")

# imaginile de testare
image_folder = 'E:/sample_images/ROC_batch_2b'  
image_list = []
# Parcurge toate subfolderele din directorul specificat
# root = calea folderului curent
# _ = lista cu subfoldere
# files = lista cu fisierele din acel folder
for root, _, files in os.walk(image_folder):
    # itereaza prin toate fisierele gasite
    for file in files: 
        # daca are extensia .png
        if file.endswith('.png'):
            # construieste calea absoluta catre fisier si o adauga in lista
            image_list.append(os.path.join(root, file))


config_path = 'mmocr/configs/textrecog/crnn/crnn_chip_finetune.py'  # <- fisierul de configurare folosit la antrenare
checkpoint_path = 'mmocr/work_dirs/crnn_chip_training/crnn_chip_training/epoch_150.pth'  # <- incarca greutatile antrenate
output_file = 'mmocr/work_dirs/crnn_chip_training/predictions/predicted_texts_batch_2b.json'  # <- unde să fie salvat output-ul

# rulare functie
save_predictions(config_path, checkpoint_path, image_list, output_file)  # va genera fisierul JSON cu predictii

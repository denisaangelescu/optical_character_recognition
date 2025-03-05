# generare fisiere json
# antrenare (primul desi mic - overfitting? -> augumentare = creste nr de date de antrenament) 
# si validare (al doilea)

# folosita pentru a lucra cu fisiere json
import json

# gt_path = locatia fisierului GT
# output_json = locatia unde va fi salvat fisierul json de iesire
gt_path = r"C:\Users\denis\Desktop\ocr\labels imagini\per branch\gt_roc_batch_2a.txt"
output_json = r"C:\Users\denis\Desktop\ocr\labels imagini\per branch\textrecog_train.json"

# lista va contine informatii despre fiecare imagine
data_list = []

# deschide fisierul in modul citire ('r')
# fiecare linie a acestui fisier contine o cale de imagine si textul asociat
with open(gt_path, 'r') as f:
    # parcurge fiecare linie a fisierului GT
    for line in f:
        # elimina spatiile de la inceput si sfarsit si imparte linie in calea imaginii si eticheta
        path, label = line.strip().split('\t')  # fisier tab-separated
        label = label.replace(" ", "")  # elimin spatiile din label -> limitare data din cauza dictionarului
        # creeaza un dictionar care contine calea catre imagine si lista cu un singur element unde "text" = eticheta
        entry = {
            "img_path": path,  
            "instances": [{"text": label}]
        }
        # adauga informatia in lista
        data_list.append(entry)

# creeaza structura finala json
final_data = {
    "metainfo": {}, # dictionar gol
    "data_list": data_list # lista
}

# deschide output_json pentru scriere
with open(output_json, 'w') as out:
    # scrie continutul final_data in fisier cu indentare de 4 spatii
    json.dump(final_data, out, indent=4)

# afiseaza locatia fisierului generat
print(f"Fisier JSON generat la: {output_json}")

# Inferenta in deep learning inseamna 
# procesul de aplicare a unui model antrenat pe date noi pentru a face predictii


# MMOCRInferencer ruleaza inferenta pe imagini
from mmocr.apis import MMOCRInferencer

# Ce model vreau:
#  - det='dbnetpp' pentru detectare text
#  - rec='svtr-small' pentru recunoastere text
#  - sau combinatie: det='dbnetpp', rec='svtr-small'
# se creeaza un obiect infer care foloseste MMOCRInferencer
infer = MMOCRInferencer(
    det='dbnetpp',       
    rec='svtr-small',    
    device='cpu'         # Folosesc CPU
)

# Apelează inferenta pe o imagine din demo
result = infer(
    'demo/chip3.jpg',
    save_vis=True,       # Salvează o imagine cu predicțiile desenate
    return_vis=True      # Intoarce imaginea prelucrata in "result"
)

# Afiseaza predictiile în consola
print(result['predictions'])

# Sa vizualizez imaginea marcata
import matplotlib.pyplot as plt

plt.imshow(result['visualization'][0]) # afiseaza imaginea; care imagine? prima rezultata cu textul detectat
plt.show() #afiseaza fereastra grafica cu vizualizarea

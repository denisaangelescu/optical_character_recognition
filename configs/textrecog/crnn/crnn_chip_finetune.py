# fisier ce defineste configuratia pentru antrenare si validare


_base_ = [
    '../_base_/default_runtime.py', # seteaza logarea, salvarea modelului
    '../_base_/schedules/schedule_adadelta_5e.py', # defineste optimizer si programul de invatare (Adadelta, 5 epoci)
    '_base_crnn_mini-vgg.py', # arhitectura
]

# incarcator de date pentru antrenare 
train_dataloader = dict(
    batch_size=8, # 8 imagini per batch
    num_workers=4, # 4 fire de incarcare paralela
    persistent_workers=True, # mentine workers activi
    sampler=dict(type='DefaultSampler', shuffle=True), # amesteca imaginile la fiecare epoca
    dataset=dict(
        type='OCRDataset',
        data_root='E:/sample_images/ROC_batch_2a', # director cu imagini de antrenament
        ann_file='C:/Users/denis/Desktop/ocr/labels imagini/per branch/textrecog_train.json', # fisier json cu etichete
        pipeline=_base_.train_pipeline))  # pipeline de preprocesare pentru antrenare

# incarcator de date pentru validare
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False, # nu se elimina ultimul batch, chiar daca e incomplet (e vorba de batch_size)
    sampler=dict(type='DefaultSampler', shuffle=False), # nu amesteca imaginile la validare
    dataset=dict(
        type='OCRDataset',
        data_root='E:/sample_images/ROC_batch_2b',
        ann_file='C:/Users/denis/Desktop/ocr/labels imagini/per branch/textrecog_val.json',
        pipeline=_base_.test_pipeline))

# aceleasi caracteristici
test_dataloader = val_dataloader

# foloseste optimizerul Adadelta (algoritm care ajusteaza parametrii: greutati, bias, ca sa minimizeze pierderea)
# greutatile se actualizeaza (backpropagation)
# se adapteaza automat la gradientul pierderii = derivata functiei de pierdere fata de fiecare parametru al modelului
# adica cat de mult trebuie modificat fiecare parametru pentru a reduce pierderea
optim_wrapper = dict(
    optimizer=dict(type='Adadelta', lr=0.0001)
)

# antreneaza modelul timp de 250 epoci
# evalueaza la fiecare epoca
# va folosi val_evaluator
train_cfg = dict(max_epochs=250, val_interval=1)

# learning rate constant
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
# lr scade treptat pe baza unei functii cos
# T_max -> nr epoci
# eta_min -> lr min
# param_scheduler = [
#     dict(type='CosineAnnealingLR', T_max=100, eta_min=1e-5)
# ]
# e mai nasol decat ea constanta


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1), # afiseaza loguri la fiecare iteratie
    # salveaza un checkpoint la fiecare epoca
    # modele salvate in out_dir
    checkpoint=dict(type='CheckpointHook', interval=1, out_dir='work_dirs/crnn_chip_training/')
)

# nu merge (trebuia sa vizualizez rezultatele ocr pe imaginile de test)
# visualizer = dict(
#     type='TextRecogLocalVisualizer',
#     vis_backends=[dict(type='LocalVisBackend')],
#     name='visualizer',
#     save_dir='work_dirs/crnn_chip_training/predictions'
# )

# fol pt a evalua performanta pe datele de validare
val_evaluator = dict(dataset_prefixes=['ROC_CHIP'])

# pt testare
test_evaluator = val_evaluator

# incarca un model preantrenat -> antrenat pe MJSynth
load_from = 'checkpoints/crnn_mini-vgg_5e_mj.pth'

# director unde se salveaza modelele antrenate
work_dir = 'work_dirs/crnn_chip_training/'

# daca am oprit simularea, nu pot sa dau resume de la o "epoca" anume
resume = None



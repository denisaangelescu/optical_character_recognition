# fisier de configurare -> arhitectura

# dictionar
dictionary = dict(
    type='Dictionary', # dictionar de caractere
    dict_file='{{ fileDirname }}/../../../dicts/lower_english_digits.txt', # calea catre fisier
    # am modif in dictionar sa am litere mari si cifre
    # nu l am folosit pe cel cu spatiu pt ca aparea eroare: 
    # modelul din checkpoints folosea dictionarul fara spatiu, deci si eu trebuia sa folosesc fara spatiu
    # inclusiv sa am in fisierul json fara spatiu
    with_padding=False # adauga un caracter special folosit pentru a completa secventele de text la aceeasi lungime
    )

# modelul
model = dict(
    type='CRNN',
    preprocessor=None, # nu se aplica o preprocesare suplimentara inainte de a intra in model (singura train_pipeline)
    # MiniVGG = arhitectura CNN mai mica, folosita pentru a extrage caracteristici vizuale din imagini
    # type specifica reteaua convolutionala utilizata
    # nu foloseste Leaky ReLU, ci ReLU normal ca functie de activare
    # modelul primeste imagini grayscale (un singur canal)
    backbone=dict(type='MiniVGG', leaky_relu=False, input_channels=1), 
    encoder=None, # nu e nevoie de encoder, caci CNN extrage caract., iar RNN proceseaza secventele de caractere
    decoder=dict( # decoder = reteaua recurenta RNN
        type='CRNNDecoder', # decodor tip CRNN
        in_channels=512, # modelul CNN produce 512 canale caracteristici, transmise la RNN
        rnn_flag=True, # activeaza partea RNN
        # max_seq_len=20, # modelul poate recunoaste pana la 20 de caractere dintr o imagine
        # module_loss=dict(type='CTCModuleLoss', letter_case='unchanged'),
        # CTC (Connectionist Temporal Classification)
        # CTC permite antrenarea modelului fara a avea nevoie de alinierea exacta intre imagine si text
        # type -> utilizeaza CTC Loss pentru antrenare
        # invata doar caractere mari
        module_loss=dict(type='CTCModuleLoss', letter_case='upper'), 
        # type -> foloseste un postprocesor pentru decodarea secventei de caractere
        # beam_width compara mai multe variante de secvente si alege pe cea mai buna
        postprocessor=dict(type='CTCPostProcessor')
        #, beam_width=10), # Beam search activat
        # dictionar definit mai sus
        dictionary=dictionary),
    # normalizeaza imaginile, standardizeaza intensitatea pixelilor (0 - 255) 
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127]))

file_client_args = dict(backend='disk') # date citite de pe hard disk-ul local

# pipeline de preprocesare
train_pipeline = [
    dict(
        type='LoadImageFromFile', # incarca imaginile din fisiere
        color_type='grayscale', # conversie alb-negru (asa am vazut ca e compatibil modelul)
        ignore_empty=True, # daca exista o imagine corupta sau lipsa, nu opreste antrenarea
        min_size=2), # trebuie sa aiba minim 2x2px, altfel este ignorata
    # incarca etichetele OCR asociate imaginilor
    # with_text -> trebuie sa incarce textul din imagine    
    dict(type='LoadOCRAnnotations', with_text=True), 
    # dict(type='Resize', scale=(100, 32), keep_ratio=False),
    # Resize -> dimensiune fixa
    # redimensioneaza la 800x32px fiecare imagine
    # 32px este standard in CRNN
    # keep_ratio -> pastreaza proportiile originale ale textului, prevenind distorsiuni
    dict(type='Resize', scale=(800, 32), keep_ratio=True),
    # dict(
    #     type='RandomRotate',
    #     angle=5,  # rotire aleatorie între -5grd si 5grd
    #     prob=0.2  # 50% sansa
    # ),
    #
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    # dict(
    #    type='RescaleToHeight',
    #    height=32,
    #    min_width=32,
    #    max_width=None,
    #    width_divisor=16),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
     dict(
        type='Resize',
        scale=(800, 32),  # Resize constant pentru testare
        keep_ratio=True),

    dict(type='LoadOCRAnnotations', with_text=True),
    # pregateste tot pentru inferenta
    # img_path -> calea imaginii originale
    # ori_shape -> dimensiunea originala a imaginii
    # img_shape -> dimensiunea imaginii dupa redimensionare
    # valid_ratio -> raportul dintre dimensiunea originala și cea scalata
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# test time augumentation
tta_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='TestTimeAug', # permite modelului sa ruleze inferenta pe mai multe versiuni ale aceleiasi imagini
        # si se combina rezultatele
        transforms=[
            [
                # urm. 3: daca imaginea este mai inalta decat lata se aplica 3 rotatii posibile
                # k=0 fara rotatie
                # k=1 90 grade
                # k=3 270 grade
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=0, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=1, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=3, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            #[
            #    dict(
            #        type='RescaleToHeight',
            #        height=32,
            #        min_width=32,
            #        max_width=None,
            #        width_divisor=16)
            #],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [
                dict(
                    type='Resize',
                    scale=(800, 32),  # Adăugat resize consistent
                    keep_ratio=True
                )
            ],

            [dict(type='LoadOCRAnnotations', with_text=True)],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]
        ])
]

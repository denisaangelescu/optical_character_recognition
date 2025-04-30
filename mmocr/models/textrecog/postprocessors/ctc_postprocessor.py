# clasa ce implementeaza postprocesarea pentru decodarea CTC folosind Beam Search


"""
import math # functii matematice
from typing import Sequence, Tuple # tipuri de date pentru indicarea formatului returnat

# functii pentru operatiuni tensoriale
import torch 
import torch.nn.functional as F 


from mmocr.registry import MODELS # putem accesa modele ocr definite in mmocr
from mmocr.structures import TextRecogDataSample # defineste structura datelor
from .base import BaseTextRecogPostprocessor # clasa de baza pentru postprocesare
import numpy as np # operatiuni matematice rapide

# TODO support beam search
@MODELS.register_module() # inregistreaza clasa 
class CTCPostProcessor(BaseTextRecogPostprocessor): # extinde BaseTextRecogPostprocessor
    #PostProcessor for CTC.

    def __init__(self, beam_width=10, **kwargs): # constructor
        # keyword arguments este un mecanism care permite metodei sa primeasca orice parametru suplimentar 
        # fara sa l specifice explicit
        super().__init__(**kwargs) # apeleaza constructorul clasei de baza
        # beam_width compara mai multe variante de secvente si alege pe cea mai buna
        self.beam_width = beam_width 

    # gaseste secvente mai bune decat greedy decoding
    def beam_search_decoder(self, probs, beam_width):
        
        #Beam Search Decoder pentru CTC.
        
        T, C = probs.shape  # T = lung. secv. de iesire; C = nr. total de caractere posibile
        beam = [([], 0)]  # (secventa goala, scor log-probabilitate)

        for t in range(T): # parcurgem fiecare pas de timp
            new_beam = [] # noile secvente candidate
            for seq, score in beam: # pt fiecare secventa candidat
                for c in range(C): #pt fiecare caracter posibil
                    new_seq = seq + [c] # adaug la secventa 
                    new_score = score + np.log(probs[t, c] + 1e-8)  # adaug probabilitatea; evitam log(0)
                    new_beam.append((new_seq, new_score))

            # alegem top "beam_width" secvente cu scor maxim (sortate descrescator dupa scor)
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        # best_seq -> secv. de caractere prezisa cu cea mai mare probabilitate
        # best_score -> lista de scoruri
        # o alegem pe cea mai buna
        best_seq, best_score = beam[0]
        return best_seq, [1.0] * len(best_seq)  # Nu avem scor per caracter
        

    def get_single_prediction(self, probs: torch.Tensor,
                              data_sample: TextRecogDataSample
                              ) -> Tuple[Sequence[int], Sequence[float]]:
        # probs -> contine probabilitatile fiecarui caracter pentru fiecare pas de timp
        # data_sample -> contine informatii despre imaginea procesata
        # index -> secventa de caractere prezisa
        # score -> scorurile fiecarui caracter
        # Convert the output probabilities of a single image to index and
        # score.

        # Args:
        #     probs (torch.Tensor): Character probabilities with shape
        #         :math:`(T, C)`.
        #     data_sample (TextRecogDataSample): Datasample of an image.

        # Returns:
        #     tuple(list[int], list[float]): index and score.
        
        feat_len = probs.size(0) # extrage lungimea secventei de timp (returneaza T)
        
        # valid_ratio indica cat din secventa generata este valida
        valid_ratio = data_sample.get('valid_ratio', 1) # se ia valoarea respectiva sau daca nu exista, 1
        # feat_len * valid_ratio -> cat din feat_len este valid
        # math.ceil() -> rotunjeste in sus la un nr intreg
        # decode_len nu depaseste totusi feat_len
        decode_len = min(feat_len, math.ceil(feat_len * valid_ratio)) # lungimea reala a secventei de iesire

        # Beam Search Decoding
        # converteste din tensor in array
        index, score = self.beam_search_decoder(probs[:decode_len].cpu().numpy(), self.beam_width)
        return index, score

    def __call__(
        self, outputs: torch.Tensor, # matricea de probabilitati pt fiecare caracter
        data_samples: Sequence[TextRecogDataSample] # lista de exemple de date, fiecare obiect contine info despre 
        # imagine si textul asociat
    ) -> Sequence[TextRecogDataSample]: # lista de obiecte TextRecogDataSample care contin textul decodat
        outputs = outputs.cpu().detach() # se muta pe cpu si nu se mai fol. backpropagation (economiseste mem si 
        # accelereaza procesarea)
        return super().__call__(outputs, data_samples) # clasa de baza are deja implementata logica de decodare
        # eu doar o extind

        """



import math
from typing import Sequence, Tuple

import torch

from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseTextRecogPostprocessor


# TODO support beam search
@MODELS.register_module()
class CTCPostProcessor(BaseTextRecogPostprocessor):
    """PostProcessor for CTC."""

    def get_single_prediction(self, probs: torch.Tensor,
                              data_sample: TextRecogDataSample
                              ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output probabilities of a single image to index and
        score.

        Args:
            probs (torch.Tensor): Character probabilities with shape
                :math:`(T, C)`.
            data_sample (TextRecogDataSample): Datasample of an image.

        Returns:
            tuple(list[int], list[float]): index and score.
        """
        feat_len = probs.size(0)
        max_value, max_idx = torch.max(probs, -1)
        valid_ratio = data_sample.get('valid_ratio', 1)
        decode_len = min(feat_len, math.ceil(feat_len * valid_ratio))
        index = []
        score = []

        prev_idx = self.dictionary.padding_idx
        for t in range(decode_len):
            tmp_value = max_idx[t].item()
            if tmp_value not in (prev_idx, *self.ignore_indexes):
                index.append(tmp_value)
                score.append(max_value[t].item())
            prev_idx = tmp_value
        return index, score

    def __call__(
        self, outputs: torch.Tensor,
        data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        outputs = outputs.cpu().detach()
        return super().__call__(outputs, data_samples)
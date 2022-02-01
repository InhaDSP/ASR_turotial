
#-*- coding: utf-8 -*-

import json
import csv
from token_list import libri_token

def load_label_tokenList(token_list):
    txt2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    idx2txt = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}

    for i, char in enumerate(token_list):
        txt2idx[char] = i + 4
        idx2txt[i + 4] = char

    return txt2idx, idx2txt 
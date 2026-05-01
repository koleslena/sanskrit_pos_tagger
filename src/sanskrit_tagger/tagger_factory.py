import os
from os.path import join
import pickle
import json

from sanskrit_tagger.pos_tagger import POSTagger
from sanskrit_tagger.segmenter import Segmenter

_data_path = 'data'
_this_dir, _ = os.path.split(__file__)

def get_pos_tagger(model, **kwargs):
    p_list = [p.shape for p in model.parameters()]
    vocab_size, labels_num = p_list[0][0], p_list[-1][0]

    with open(join(_this_dir, _data_path, f'char2id_{vocab_size}_{labels_num}.dat'), 'rb') as file:
        loaded_dict = pickle.load(file)

    with open(join(_this_dir, _data_path, f'unique_tags_{vocab_size}_{labels_num}.dat'), 'rb') as f:
        loaded_array = pickle.load(f)

    return POSTagger(model, loaded_dict, loaded_array, **kwargs)

def _load_vocab(path_char2id):
    with open(path_char2id, 'r', encoding='utf-8') as f:
        char2id = json.load(f)
        
    return char2id

def get_segmenter(model, **kwargs):
    char2id = _load_vocab(join(_this_dir, _data_path, 'char2id.json'))
    return Segmenter(model, char2id, **kwargs)


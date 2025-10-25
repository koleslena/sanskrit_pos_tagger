import os
from os.path import join
import pickle

from sanskrit_tagger.pos_tagger import POSTagger

_data_path = 'data'
_this_dir, _ = os.path.split(__file__)

def get_pos_tagger(model, vocab_size, labels_num, **kwargs):
    with open(join(_this_dir, _data_path, f'char2id_{vocab_size}_{labels_num}.dat'), 'rb') as file:
        loaded_dict = pickle.load(file)

    with open(join(_this_dir, _data_path, f'unique_tags_{vocab_size}_{labels_num}.dat'), 'rb') as f:
        loaded_array = pickle.load(f)

    return POSTagger(model, loaded_dict, loaded_array, **kwargs)
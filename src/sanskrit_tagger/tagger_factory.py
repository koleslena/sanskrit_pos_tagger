from sanskrit_tagger.pos_tagger import POSTagger
from sanskrit_tagger.segmenter import Segmenter

def get_pos_tagger(model, **kwargs):
    return POSTagger(model, model.char2id, model.tags, **kwargs)

def get_segmenter(model, **kwargs):
    return Segmenter(model, model.char2id, **kwargs)


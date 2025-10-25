from sanskrit_tagger.tagger_factory import get_pos_tagger

def test_get_pos_tagger():
    assert get_pos_tagger(None, 42, 734) is not None

def test_pos_tagger_params():
    tagger = get_pos_tagger(None, 42, 734)
    assert len(tagger.char2id) == 42
    assert len(tagger.id2label) == 734

def test_pos_tagger_kwargs():
    tagger = get_pos_tagger(None, 42, 734, max_sent_len=1000)
    assert tagger.max_sent_len == 1000
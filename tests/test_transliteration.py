import pytest
from sanskrit_tagger.transliteration import normalize_to_slp1, slp_to_deva, slp_to_iast



@pytest.mark.parametrize("input_text, expected_slp1", [
    # --- Деванагари ---
    ("गच्छति", "gacCati"),
    ("देव", "deva"),
    ("अश्व", "aSva"),
    ("कतमत्-कतमद्", "katamat-katamad"),
    
    # --- IAST (с диакритикой) ---
    ("gacchati", "gacCati"),
    ("devābhyām", "devAByAm"),
    ("ṛṣi", "fzi"),
    ("kṛṣṇa", "kfzRa"),
    ("laṅghanaṁ kiṁsvillaṅghanīyāśca", "laNGanaM kiMsvillaNGanIyASca"),
    ("agnim saṅgṛhya jalam pibati", "agnim saNgfhya jalam pibati"),
    
    # --- Harvard-Kyoto (HK) ---
    ("bhavati", "Bavati"),  # bh -> B
    ("khadati", "Kadati"),  # kh -> K
    ("phalam", "Palam"),    # ph -> P
    ("divau", "divO"),    # au -> O
    ("guNa", "guRa"),   # R -> N
    
    # --- SLP1 (Без изменений) ---
    ("gaccati", "gaccati"),
    ("Bavati", "Bavati"),
    ("fzi", "fzi"),
    ("divO", "divO"),
    ("guRa", "guRa"),
    
    # --- Пустые и краевые случаи ---
    ("", ""),
    ("   ", ""),
])

def test_normalize_to_slp1(input_text, expected_slp1):
    """Тестирование корректности транслитерации в формат SLP1."""
    assert normalize_to_slp1(input_text) == expected_slp1

def test_slp1_specific_markers():
    """Отдельный тест на специфичные для SLP1 гласные."""
    # f = ṛ, F = ṝ
    assert normalize_to_slp1("fzi") == "fzi"
    assert normalize_to_slp1("pitFn") == "pitFn" # pitṝn
    assert normalize_to_slp1("ṛṣi") == "fzi"
    assert normalize_to_slp1("ṛtu") == "ftu"
    assert normalize_to_slp1("pitṝn") == "pitFn" # pitṝn

def test_slp_to_deva():
    assert " ".join([slp_to_deva(ans) for ans in ["he katamat,he katamad", "he katame", "he katamAni"]]) == "हे कतमत्,हे कतमद् हे कतमे हे कतमानि"


def test_slp_to_iast():
    assert " ".join([slp_to_iast(ans) for ans in ["he katamat,he katamad", "he katame", "he katamAni"]]) == "he katamat,he katamad he katame he katamāni"
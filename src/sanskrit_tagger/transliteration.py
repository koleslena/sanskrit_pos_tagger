from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def slp_to_iast(text: str) -> str:
    return transliterate(text, sanscript.SLP1, sanscript.IAST)

def slp_to_deva(text: str) -> str:
    return transliterate(text, sanscript.SLP1, sanscript.DEVANAGARI)

def is_deva(text):
    return schema(text) == sanscript.DEVANAGARI

def normalize_result(res):
    return slp_to_deva(res) if is_deva(res) else slp_to_iast(res)

def schema(text):
    # R как ретрофлексная n 
    slp1_markers = ["aR", "AR", "iR", "IR", "uR", "UR", "oR", "OR", "eR", "ER", "Ra", "RA", "Ri", "RI", "Ru", "RU", "Ro", "RO", "Re", "RE"]
    
    # Если есть f, x, w, q или F, X, Y, O, E, W, Q — это точно SLP1, возвращаем как есть
    if any(m in text for m in "fFxXYOEwWqQ") or any(m in text for m in slp1_markers):
        return sanscript.SLP1 

    hk_markers = ["kh", "gh", "ch", "jh", "th", "dh", "Th", "Dh", "ph", "bh", "au", "ai", "J", "RR"]

    # Пытаемся определить: если есть символы деванагари
    if any('\u0900' <= char <= '\u097F' for char in text):
        return sanscript.DEVANAGARI
    # Если есть диакритика (ā, ī, ū, ṃ, ḥ, ṭ, ḍ, ṇ, ś, ṣ, ṅ, ṛ, ñ, ṝ) — скорее всего IAST
    elif any(char in "āīūṃḥṭḍṇśṣṁṅṛñṝ" for char in text.lower()):
        return sanscript.IAST
    # Если есть характерные придыхания 'h' — скорее всего это HK
    elif any(m in text for m in hk_markers):
        return sanscript.HK
    
    # По умолчанию HK (так как S и z, поменяны в HK и SLP1 и нет возможности их отличить, но чаще все-таки люди используют HK)
    return sanscript.HK

def normalize_to_slp1(text: str) -> str:
    """
    Определяет (примерно) схему письма и конвертирует её в SLP1.
    Список схем, которые мы поддерживаем DEVANAGARI, IAST, HK (Harvard-Kyoto), SLP1
    """
    if not text:
        return ""
    
    text = text.strip()
    text = text.replace("ṁ", "ṃ")

    source_scheme = schema(text)
    print(source_scheme)

    if source_scheme == sanscript.SLP1:
        return text
    
    return transliterate(text, source_scheme, sanscript.SLP1)
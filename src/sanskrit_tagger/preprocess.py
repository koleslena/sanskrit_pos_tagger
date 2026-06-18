import re

MIN_WORDS = 4

def clean_text(text):
    if not text:
        return ''

    text = text.replace('-', '')
    # ШАГ 0: Бесследно удаляем скрытые непечатные символы и zero-width костыли
    # \u2000-\u200f — это блок невидимых пробелов, ZWNJ (\u200c) и ZWJ (\u200d)
    # \ufeff — BOM-маркер
    invisible_junk = r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f\u2000-\u200f\u202a-\u202f\ufeff]'
    text = re.sub(invisible_junk, '', text)

    # 1. Заменяем все варианты вертикальных черт (|, ||) и стандартный пунктуационный мусор на пробелы
    # Сохраняем буквы всех алфавитов (включая диакритику IAST и Деванагари)
    # Убираем: |, ||, |, ‖, ., ,, !, ?, ;, :, \, /, *, etc.
    cleaned_text = re.sub(r'[\d\|\|।॥\.,!\?;\::\"\(\)\[\]\{\}«»„“"”]', ' ', text)

    cleaned_text = cleaned_text.replace('saNgf', 'saMgf').replace('saṅgṛ', 'saṃgṛ')

    return cleaned_text


def preprocess_sanskrit_text(text):
    if not text:
        return []

    cleaned_text = clean_text(text)
        
    # 2. Разбиваем текст на первичные строки по символам переноса строки
    raw_lines = cleaned_text.splitlines()
    
    final_lines = []
    
    for line in raw_lines:
        # Разбиваем строку на отдельные слова и убираем лишние пробелы
        words = [w.strip() for w in line.split() if w.strip()]
        
        if not words:
            continue
            
        # Если в строке MIN_WORDS или меньше слов, оставляем как есть
        if len(words) <= MIN_WORDS:
            final_lines.append(" ".join(words))
            continue
            
        # 3. Если слов больше MIN_WORDS, запускаем алгоритм умной нарезки
        i = 0
        total_words = len(words)
        
        while i < total_words:
            remaining_words = total_words - i
            
            # Если осталось от MIN_WORDS + 1 до MIN_WORDS + 2 слов, забираем их целиком в одну строку,
            # чтобы на следующем шаге не остался хвост из 1 или 2 слов.
            if MIN_WORDS + 1 <= remaining_words <= MIN_WORDS + 2:
                chunk = words[i : i + remaining_words]
                final_lines.append(" ".join(chunk))
                break
                
            # Стандартный шаг: берем максимум MIN_WORDS слов
            chunk_size = min(MIN_WORDS, remaining_words)
            chunk = words[i : i + chunk_size]
            final_lines.append(" ".join(chunk))
            
            i += chunk_size

    return final_lines

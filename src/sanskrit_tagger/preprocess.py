import re

def preprocess_sanskrit_text(text):
    if not text:
        return []

    # 1. Заменяем все варианты вертикальных черт (|, ||) и стандартный пунктуационный мусор на пробелы
    # Сохраняем буквы всех алфавитов (включая диакритику IAST и Деванагари)
    # Убираем: |, ||, |, ‖, ., ,, !, ?, ;, :, \, /, *, etc.
    cleaned_text = re.sub(r'[\d\|\|।॥\.,!\?;\::\-\"\(\)\[\]\{\}«»„“"”]', ' ', text)
    
    # 2. Разбиваем текст на первичные строки по символам переноса строки
    raw_lines = cleaned_text.splitlines()
    
    final_lines = []
    
    for line in raw_lines:
        # Разбиваем строку на отдельные слова и убираем лишние пробелы
        words = [w.strip() for w in line.split() if w.strip()]
        
        if not words:
            continue
            
        # Если в строке 5 или меньше слов, оставляем как есть
        if len(words) <= 5:
            final_lines.append(" ".join(words))
            continue
            
        # 3. Если слов больше 5, запускаем алгоритм умной нарезки
        i = 0
        total_words = len(words)
        
        while i < total_words:
            remaining_words = total_words - i
            
            # Если осталось от 6 до 7 слов, забираем их целиком в одну строку,
            # чтобы на следующем шаге не остался хвост из 1 или 2 слов.
            if 6 <= remaining_words <= 7:
                chunk = words[i : i + remaining_words]
                final_lines.append(" ".join(chunk))
                break
                
            # Стандартный шаг: берем максимум 5 слов
            chunk_size = min(5, remaining_words)
            chunk = words[i : i + chunk_size]
            final_lines.append(" ".join(chunk))
            
            i += chunk_size

    return final_lines

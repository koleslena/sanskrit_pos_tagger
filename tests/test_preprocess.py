import pytest
# Предполагается, что функция находится в файле sanskrit_util.py
from sanskrit_tagger.preprocess import preprocess_sanskrit_text


def test_empty_and_none_input():
    """Проверяем обработку пустых строк и None."""
    assert preprocess_sanskrit_text("") == []
    assert preprocess_sanskrit_text(None) == []


def test_punctuation_and_danda_removal():
    """Проверяем, что удаляются все знаки препинания, включая одинарные и двойные данды,
    """
    text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    result = preprocess_sanskrit_text(text)

    # Должна получиться одна строка, так как слов всего 8, но после удаления
    # знаков пунктуации сработает разбиение (8 слов -> 5 + 3)
    assert len(result) == 2

    # Проверяем, что в тексте не осталось вертикальных черт и знаков препинания
    combined_result = " ".join(result)
    assert "।" not in combined_result
    assert "॥" not in combined_result
    # Проверяем сохранность диакритики и кодировки
    assert "धर्मक्षेत्रे" in combined_result
    assert "कुरुक्षेत्रे" in combined_result


@pytest.mark.parametrize("word_count, expected_chunks", [
    (3, [3]),
    (5, [5]),
    (6, [6]),
    (7, [7]),
    (8, [5, 3]),
    (9, [5, 4]),
    (10, [5, 5]),
    (11, [5, 6]),
    (12, [5, 7]),
    (13, [5, 5, 3])
])
def test_smart_word_splitting_boundaries(word_count, expected_chunks):
    """Тест без фикстуры subtests. 
    Каждое граничное условие проверяется изолированно.
    """
    # Генерируем строку с точным количеством слов
    words_list = [f"word{i}" for i in range(1, word_count + 1)]
    input_text = " ".join(words_list)
    
    result = preprocess_sanskrit_text(input_text)
    
    # 1. Проверяем общее количество получившихся строк
    assert len(result) == len(expected_chunks), \
        f"Для {word_count} слов ожидали {len(expected_chunks)} куска(ов), но получили {len(result)}"
    
    # 2. Проверяем длину каждой строки по отдельности
    for idx, expected_len in enumerate(expected_chunks):
        actual_len = len(result[idx].split())
        assert actual_len == expected_len, \
            f"Внутри теста на {word_count} слов: в куске №{idx} ожидали длину {expected_len}, а получили {actual_len}"


def test_multiline_input_processing():
    """Проверяем, что исходные переводы строк корректно обрабатываются изолированно."""
    text = """
    śravaṇaṁ kīrtanaṁ viṣṇoḥ
    smaranaṁ pāda-sevanam | arcanaṁ vandanaṁ dāsyam
    """
    result = preprocess_sanskrit_text(text)

    # Первая строка: 3 слова -> остается 3
    # Вторая строка: 6 слов -> должна остаться одной строкой из 6 слов (вместо 5 + 1)
    assert len(result) == 2
    assert len(result[0].split()) == 3
    assert len(result[1].split()) == 6


def test_extra_spaces_and_newlines():
    """Проверяем, что лишние пробелы, табы и пустые строки не ломают логику подсчета."""
    text = "   pad1    pad2\tpad3   \n\n   pad4   |   pad5   "
    result = preprocess_sanskrit_text(text)

    # Должно получиться две строки (3 слова и 2 слова)
    assert len(result) == 2
    assert result[0] == "pad pad pad"
    assert result[1] == "pad pad"
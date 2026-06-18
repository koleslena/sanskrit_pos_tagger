
# 🕉️ Sanskrit Tagger

Вспомогательная библиотека для сегментации и морфологического теггинга предложений на санскрите с использованием предобученных моделей.

## Основные возможности:

- **Сегментация текста:** Снятие сандхи, разделение текста на слова и основы в сложных словах.

- **Морфологический анализ:** Определение части речи (POS), падежа, рода, числа и глагольных форм.

- **Factory-интерфейс:** Удобное создание моделей одной функцией.

- **Поддержка IAST, HK, Devanagari:**  Работает с латинской транслитерацией санскрита.

## 🚀 Быстрый старт (Quick Start)

```bash
pip install sanskrit_tagger
```

### 📥 Загрузка моделей

Модели обучались на корпусе санскритских текстов и доступны напрямую через `torch.hub`. Вы можете выбрать одну из архитектур:

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# загрузка модели классификации
model = torch.hub.load('koleslena/sanskrit_nlp_models', 'pos_tagger_model', device=device, force_reload=True)

# загрузка модели сегментации
model_segmenter = torch.hub.load('koleslena/sanskrit_nlp_models', 'segmenter_model', device=device, force_reload=True)
```

```python
from sanskrit_tagger.tagger_factory import get_pos_tagger, get_segmenter

# Создание сегментатора
segmenter = get_segmenter(model_segmenter, device=device)

test_splitter_sentences = ["agnihotrātsamutthāya harṣeṇa mahatānvitā",
                            "uvāca cainaṁ varadā vacanaṁ pārthivaṁ tadā"]

# Получение разделения
segmented = segmenter(test_splitter_sentences)

for sent, predicted in zip(test_splitter_sentences, segmented):
    print(f"Input:  {sent}")
    print(f"Output: {predicted}")

# Создание теггера 
pos_tagger = get_pos_tagger(model, device=device)

# Текст должен быть разбит на слова
sentence = " ".join(segmented)

# Получение результата
for sent_tokens, sent_tags in zip(sentence.split(), pos_tagger(sentence)):
    print(' '.join('{}-{}'.format(tok, tag) for tok, tag in zip(sent_tokens, sent_tags)))

```

## 📊 Пример вывода (Output Example)

Библиотека возвращает детальные теги для каждого токена:
```
atha-ADV kanyā-NOUN Gen Fem Sing pradāne-NOUN Loc Neut Sing...
```

## 🛠 Технические подробности

Библиотека инкапсулирует логику предобработки и пост-обработки векторов предсказаний моделей, позволяя сосредоточиться на лингвистическом анализе, а не на тензорах.


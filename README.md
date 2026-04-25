
# 🕉️ Sanskrit Tagger

Вспомогательная библиотека для морфологического теггинга предложений на санскрите с использованием предобученных моделей классификации.

## Основные возможности:

- **Морфологический анализ:** Определение части речи (POS), падежа, рода, числа и глагольных форм.

- **Factory-интерфейс:** Удобное создание теггера одной функцией.

- **Поддержка IAST:**  Работает с латинской транслитерацией санскрита.

## 🚀 Быстрый старт (Quick Start)

```bash
pip install sanskrit_tagger
```

### 📥 Загрузка моделей

Модели обучались на корпусе санскритских текстов и доступны напрямую через `torch.hub`. Вы можете выбрать одну из архитектур:

```python
import torch

# Загрузка CNN модели (устаревшая)
model = torch.hub.load('koleslena/sanskrit_nlp_models', 'cnn_full_pos_tagger_model')

# загрузка BiLSTM модели (более точная на длинных контекстах)
model = torch.hub.load('koleslena/sanskrit_nlp_models', 'bilstm_full_pos_tagger_model')

# Извлекаем размерности словарей прямо из параметров модели
p_list = [p.shape for p in model.parameters()]
chars_dim, tags_dim = p_list[0][0], p_list[-1][0]

```

```python
from sanskrit_tagger.tagger_factory import get_pos_tagger

# Создание теггера с имеющимися параметрами (модель и словари символов/тегов)
pos_tagger = get_pos_tagger(model, chars_dim, tags_dim, max_sent_len=1000)

# Текст должен быть разбит на слова
sentences = [
    'atha kanyā pradāne sa tam eva arthaṁ vicintayan',
]

# Получение результата
for sent_tags in pos_tagger(sentences):
    print(sent_tags)
```

## 📊 Пример вывода (Output Example)

Библиотека возвращает детальные теги для каждого токена:
```
atha-ADV kanyā-NOUN Gen Fem Sing pradāne-NOUN Loc Neut Sing...
```

## 🛠 Технические подробности

Библиотека инкапсулирует логику предобработки и пост-обработки векторов предсказаний моделей, позволяя сосредоточиться на лингвистическом анализе, а не на тензорах.


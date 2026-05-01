import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sanskrit_tagger.device_util import get_device, copy_data_to_device
from sanskrit_tagger.transliteration import normalize_to_slp1

class POSTagger:
    def __init__(self, model, char2id, id2label, device=None, batch_size=64):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.device = device
        self.batch_size = batch_size

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 1. Нормализация
        sentences = [normalize_to_slp1(sent) for sent in sentences]
        
        # 2. Токенизация (разбиваем на слова)
        tokenized_corpus = [sent.split() for sent in sentences]
        
        # 3. Считаем макс. длины как в collate_fn (только для текущего набора предложений)
        batch_max_sent_len = max(len(sent) for sent in tokenized_corpus)
        batch_max_token_len = max(max(len(word) for word in sent) for sent in tokenized_corpus)

        # 4. Формируем тензор в точности как в pos_collate_fn
        # inputs shape: (Batch, MaxSentLen, MaxTokenLen)
        inputs = torch.zeros((len(sentences), batch_max_sent_len, batch_max_token_len), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                for char_i, char in enumerate(token):
                    # Используем тот же char2id.get(char, 0) и ТОТ ЖЕ ИНДЕКС (без +1)
                    inputs[sent_i, token_i, char_i] = self.char2id.get(char, 0)

        # 5. Предикт
        # Важно: TensorDataset тут нужен только для упаковки, 
        # вторая часть (нули) просто заглушка для DataLoader
        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        
        # predicted_probs: (BatchSize, TagsN, MaxSentLen)
        predicted_probs = self._predict_with_model(dataset)  
        
        # Получаем классы (BatchSize, MaxSentLen)
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent_tokens in enumerate(tokenized_corpus):
            # Декодируем только реальные слова (отсекаем паддинг)
            sent_labels = [self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent_tokens)]]
            result.append(sent_labels)
            
        return result
    
    def _predict_with_model(self, dataset, return_labels=False):
        device = get_device(self.device)
        results_by_batch = []

        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        labels = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, total=len(dataset)/self.batch_size):
                batch_x = copy_data_to_device(batch_x, device)

                if return_labels:
                    labels.append(batch_y.numpy())

                batch_pred = self.model(batch_x)
                results_by_batch.append(batch_pred.detach().cpu().numpy())

        if return_labels:
            return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
        else:
            return np.concatenate(results_by_batch, 0)
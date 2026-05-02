import torch

from sanskrit_tagger.device_util import get_device
from sanskrit_tagger.transliteration import normalize_result, normalize_to_slp1

class Segmenter:
    def __init__(self, model, char2id, device=None, batch_size=64):
        self.model = model
        self.char2id = char2id
        self.id2char = {idx: char for char, idx in char2id.items()}
        self.device = device
        self.batch_size = batch_size

    def __call__(self, sentences):

        if isinstance(sentences, str):
            sentences = [sentences]
        
        sentences = [normalize_to_slp1(sent) for sent in sentences]

        max_sent_len = max(len(sent) for sent in sentences)
        device = get_device(self.device)

        result = []
        for sent in sentences:
            result.append(normalize_result(self._split_sentence(self.model, sent, self.char2id, self.id2char, device, max_len = max_sent_len + 10)))
        return result
    
    def _split_sentence(self, model, sentence, char_to_idx, idx_to_char, device, max_len=1500):
        SOS = char_to_idx['<SOS>']
        EOS = char_to_idx['<EOS>']

        model.eval()
        
        # 1. Подготовка входа (SLP1 -> Индексы)
        # Добавляем <EOS> 
        tokens = [char_to_idx.get(c, 0) for c in sentence] + [EOS]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1 Кодируем вход
            encoder_outputs, lengths = model.encoder(src_tensor)

            # >>> СОЗДАЕМ МАСКУ ДЛЯ ВНИМАНИЯ <<<
            # 0 - это индекс PAD. True там, где реальный символ.
            src_mask = (src_tensor != 0)

            # 2 Начальное состояние декодера
            # Вычисляем индекс последнего реального символа в последовательности
            last_idx = lengths[0] - 1
            
            # encoder_outputs имеет размер [batch=1, seq_len, hidden_dim]
            # encoder_outputs[:, last_idx, :] берет вектор [1, hidden_dim]
            # .unsqueeze(0) делает его [1, 1, hidden_dim], как ожидает LSTM
            last_hidden = encoder_outputs[:, last_idx, :].unsqueeze(0).contiguous()

            hidden_states = []
            for _ in range(model.n_layers):
                h = last_hidden.clone()
                # Создаем тензор для cell state (c) того же типа на нужном устройстве
                c = torch.zeros(1, 1, model.hidden_dim, device=device, dtype=last_hidden.dtype)
                hidden_states.append((h, c))
            
            # Начинаем с <SOS> (индекс 59)
            current_char_idx = SOS 
            result = []
            
            for _ in range(max_len):
                input_tensor = torch.LongTensor([[current_char_idx]]).to(device)
                
                # Проход через декодер с Pointer Mechanism
                log_prob, hidden_states = model.decoder(
                    input_tensor, 
                    hidden_states, 
                    encoder_outputs, 
                    src_tensor,
                    src_mask
                )
                
                # Выбираем самый вероятный символ
                current_char_idx = log_prob.argmax(1).item()
                
                # Если дошли до <EOS> (60), останавливаемся
                if current_char_idx == EOS:
                    break
                    
                result.append(idx_to_char.get(current_char_idx, '?'))
                
        return "".join(result)
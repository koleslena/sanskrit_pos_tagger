import torch

from sanskrit_tagger.device_util import get_device
from sanskrit_tagger.preprocess import preprocess_sanskrit_text
from sanskrit_tagger.transliteration import normalize_result, normalize_to_slp1

class Segmenter:
    def __init__(self, model, char2id, device=None, beam_width=5):
        self.model = model
        self.char2id = char2id
        self.id2char = {idx: char for char, idx in char2id.items()}
        self.device = get_device(device)
        self.beam_width = beam_width
        self.SOS = self.char2id['<SOS>']
        self.EOS = self.char2id['<EOS>']

    def __call__(self, sentences, beam=True, return_top=1):
        """
        sentences: список строк или одна строка
        return_top: сколько лучших вариантов возвращать (1 или больше)
        """
        if isinstance(sentences, str):
            sentences = preprocess_sanskrit_text(sentences)
        else:
            sentences = [line for sent in sentences for line in preprocess_sanskrit_text(sent)]
        
        sentences = [normalize_to_slp1(sent) for sent in sentences]

        final_results = []
        if beam:
            for sent in sentences:
                # Получаем список из beam_width гипотез
                hypotheses = self._split_sentence_beam(sent, max_len = len(sent) * 2)
                
                # Очищаем и нормализуем топ-N гипотез
                processed_variants = []
                for score, seq_ids in hypotheses[:return_top]:
                    # Декодируем, пропуская служебные символы
                    decoded = [self.id2char.get(idx, '?') for idx in seq_ids 
                            if idx not in [self.SOS, self.EOS]]
                    raw_text = "".join(decoded)
                    processed_variants.append(normalize_result(raw_text))
                
                # Если просили 1 — возвращаем строку, если больше — список
                if return_top == 1:
                    final_results.append(processed_variants[0])
                else:
                    final_results.append(processed_variants)
        else:
            max_sent_len = max(len(sent) for sent in sentences)
            for sent in sentences:
                final_results.append(normalize_result(self._split_sentence(sent, max_len = max_sent_len * 2)))
                            
        return final_results
    
    def _split_sentence(self, sentence, max_len=2500):

        self.model.eval()
        
        # 1. Подготовка входа (SLP1 -> Индексы)
        # Добавляем <EOS> 
        tokens = [self.char2id.get(c, 0) for c in sentence] + [self.EOS]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1 Кодируем вход
            encoder_outputs, lengths = self.model.encoder(src_tensor)

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
            for _ in range(self.model.n_layers_dec):
                h = last_hidden.clone()
                # Создаем тензор для cell state (c) того же типа на нужном устройстве
                c = torch.zeros(1, 1, self.model.hidden_dim, device=self.device, dtype=last_hidden.dtype)
                hidden_states.append((h, c))
            
            # Начинаем с <SOS> (индекс 59)
            current_char_idx = self.SOS 
            result = []

            encoder_projected = self.model.decoder.attention.U(encoder_outputs)
            
            for _ in range(max_len):
                input_tensor = torch.LongTensor([[current_char_idx]]).to(self.device)
                
                # Проход через декодер с Pointer Mechanism
                log_prob, hidden_states = self.model.decoder(
                    input_tensor, 
                    hidden_states, 
                    encoder_outputs, 
                    src_tensor,
                    src_mask,
                    encoder_projected=encoder_projected
                )
                
                # Выбираем самый вероятный символ
                current_char_idx = log_prob.argmax(1).item()
                
                # Если дошли до <EOS> (60), останавливаемся
                if current_char_idx == self.EOS:
                    break
                    
                result.append(self.id2char.get(current_char_idx, '?'))
                
        return "".join(result)
    
    def _split_sentence_beam(self, sentence, max_len=200):

        self.model.eval()
        
        # 1. Подготовка входа (SLP1 -> Индексы)
        # Добавляем <EOS> 
        tokens = [self.char2id.get(c, 0) for c in sentence] + [self.EOS]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # (Энкодер и подготовка)
            encoder_outputs, lengths = self.model.encoder(src_tensor)
            src_mask = (src_tensor != 0)
            encoder_projected = self.model.decoder.attention.U(encoder_outputs)
            
            # (Hidden states)
            last_idx = lengths[0] - 1
            last_hidden = encoder_outputs[:, last_idx, :].unsqueeze(0).contiguous()
            initial_hidden = [(last_hidden.clone(), torch.zeros(1, 1, self.model.hidden_dim, device=self.device)) 
                              for _ in range(self.model.n_layers)]

            # Гипотезы: (score, sequence_ids, hidden_states)
            hypotheses = [(0.0, [self.SOS], initial_hidden)]

            for _ in range(max_len):
                new_hypotheses = []
                for score, seq, h_states in hypotheses:
                    if seq[-1] == self.EOS:
                        new_hypotheses.append((score, seq, h_states))
                        continue
                    
                    input_char = torch.LongTensor([[seq[-1]]]).to(self.device)
                    log_prob, next_h = self.model.decoder(
                        input_char, h_states, encoder_outputs, 
                        src_tensor, src_mask, encoder_projected=encoder_projected
                    )

                    topv, topi = log_prob.topk(self.beam_width)
                    for i in range(self.beam_width):
                        new_hypotheses.append((score + topv[0, i].item(), seq + [topi[0, i].item()], next_h))

                hypotheses = sorted(new_hypotheses, key=lambda x: x[0], reverse=True)[:self.beam_width]
                if all(h[1][-1] == self.EOS for h in hypotheses):
                    break
            
            # Возвращаем список кортежей (score, sequence_ids)
            return [(h[0], h[1]) for h in hypotheses]
        
    
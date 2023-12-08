# Не меняйте блок кода ниже! Здесь указаны все необходимые import-ы

# __________start of block__________
import string
import os
import sys
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
# __________end of block__________

# Не меняйте блок кода ниже!
# __________start of block__________
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print('{} device is available'.format(device))
# __________end of block__________

# do not change the code in the block below
# __________start of block__________

if 'onegin.txt' not in os.listdir('../datasets/'):
    url = "https://raw.githubusercontent.com/MSUcourses/Data-Analysis-with-Python/main/Deep%20Learning/onegin_hw07.txt"
    destination = "../datasets/onegin.txt"

    response = requests.get(url)
    with open(destination, "wb") as file:
        file.write(response.content)

with open('../datasets/onegin.txt', 'r') as iofile:
    text = iofile.readlines()

text = "".join([x.replace('\t\t', '').lower() for x in
                text])  # Убираем лишние символы табуляций, приводим все буквы к нижнему регистру
# __________end of block__________

# Не меняйте блок кода ниже!
# __________start of block__________
tokens = sorted(set(text.lower())) + ['<sos>'] # Строим множество всех токенов-символов и добавляем к нему служебный токен <sos>
num_tokens = len(tokens)

assert num_tokens == 84, "Check the tokenization process"

token_to_idx = {x: idx for idx, x in enumerate(tokens)} # Строим словарь с ключами-токенами и значениями-индексами в списке токенов
idx_to_token = {idx: x for idx, x in enumerate(tokens)} # Строим обратный словарь (чтобы по индексу можно было получить токен)

assert len(tokens) == len(token_to_idx), "Mapping should be unique"

print("Seems fine!")

text_encoded = [token_to_idx[x] for x in text]
# __________end of block__________

# Не меняйте код ниже
# __________start of block__________
batch_size = 256  # Размер батча. Батч - это набор последовательностей символов.
seq_length = 100  # Максимальная длина одной последовательности символов в батче
start_column = np.zeros((batch_size, 1), dtype=int) + token_to_idx['<sos>'] # Добавляем в начало каждой строки
# технический символ - для определения начального состояния сети


def generate_chunk():
    global text_encoded, start_column, batch_size, seq_length

    start_index = np.random.randint(0, len(text_encoded) - batch_size*seq_length - 1)  # Случайным образом выбираем
    # индекс начального символа в батче
    # Строим непрерывный батч. Для этого выбираем в исходном тексте подпоследовательность, которая начинается с
    # индекса start_index и имеет размер batch_size*seq_length. Затем мы делим эту подпоследовательность на
    # batch_size последовательностей размера seq_length. Это и будет батч, матрица размера batch_size*seq_length. В
    # каждой строке матрицы будут указаны индексы
    data = np.array(text_encoded[start_index:start_index + batch_size*seq_length]).reshape((batch_size, -1))
    yield np.hstack((start_column, data))
# __________end of block__________


class CharRNNLoop(nn.Module):
    def __init__(self, num_tokens=num_tokens, emb_size=seq_length, rnn_num_units=64):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.rnn = nn.LSTM(emb_size, rnn_num_units, num_layers=2, batch_first=True, dropout=0.2)
        self.hid_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x):
        h_seq, _ = self.rnn(self.emb(x))
        next_logits = self.hid_to_logits(h_seq)
        return next_logits


model = CharRNNLoop()
opt = torch.optim.Adam(model.parameters())
criterion = nn.NLLLoss()

model.to(device)

history = []
for i in tqdm(range(10001)):

    batch_ix = torch.tensor(next(generate_chunk()), dtype=torch.int64)
    batch_ix = batch_ix.to(device)

    logits = model(batch_ix)

    # Считаем loss
    predictions_logp = F.log_softmax(logits[:, :-1], dim=-1)
    actual_next_tokens = batch_ix[:, 1:]

    loss = criterion(
        predictions_logp.contiguous().view(-1, num_tokens),
        actual_next_tokens.contiguous().view(-1)
    )

    # Обучение методов backprop
    loss.backward()
    opt.step()
    opt.zero_grad()

    # код отрисовки графика
    history.append(loss.item())
    if (i % 10000 == 0) and (i != 0):
        clear_output(True)
        plt.plot(history, label='loss')
        plt.legend()
        plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge."


def generate_sample(char_rnn, seed_phrase=None, max_length=500, temperature=1.0, device=device):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
    :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
    :param max_length: maximum output length, including seed_phrase
    :param temperature: coefficient for sampling.  higher temperature produces more chaotic outputs,
                        smaller temperature converges to the single most likely output
    '''

    if seed_phrase is not None:
        x_sequence = [token_to_idx['<sos>']] + [token_to_idx[token] for token in seed_phrase]
    else:
        x_sequence = [token_to_idx['<sos>']]

    x_sequence = torch.tensor([x_sequence], dtype=torch.int64).to(device)

    for _ in range(max_length - len(seed_phrase)):
        logits = model(x_sequence)

        p_next = F.softmax(logits / temperature, dim=-1).cpu().data.numpy()[0][-1]

        next_ix = np.random.choice(num_tokens, p=p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64).to(device)

        x_sequence = torch.cat([x_sequence, next_ix], dim=1)

    return ''.join([tokens[ix] for ix in x_sequence.cpu().data.numpy()[0]])


seed_phrase = ' мой дядя самых честных правил'

generated_phrases = [
    generate_sample(
        model,
        seed_phrase=seed_phrase,
        max_length=500,
        temperature=0.8
    ).replace('<sos>', '')
    for _ in tqdm(range(10))
]

print(generated_phrases)
print(len(generated_phrases[0]))

# do not change the code in the block below
# __________start of block__________

if 'generated_phrases' not in locals():
    raise ValueError("Please, save generated phrases to `generated_phrases` variable")

for phrase in generated_phrases:

    if not isinstance(phrase, str):
        raise ValueError("The generated phrase should be a string")

    if len(phrase) != 500:
        raise ValueError("The `generated_phrase` length should be equal to 500")

    assert all([x in set(tokens) for x in set(list(phrase))]), 'Unknown tokens detected, check your submission!'

submission_dict = {
    'token_to_idx': token_to_idx,
    'generated_phrases': generated_phrases
}

np.save('submission_dict_hw07.npy', submission_dict, allow_pickle=True)
print('File saved to `submission_dict_hw07.npy`')
# __________end of block__________
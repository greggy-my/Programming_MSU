import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output

decoder_hidden_state = np.array([7, 11, 4]).astype(float)[:, None]

plt.figure(figsize=(2, 5))
plt.pcolormesh(decoder_hidden_state)
plt.colorbar()
plt.title('Decoder state')

single_encoder_hidden_state = np.array([1, 5, 11]).astype(float)[:, None]

plt.figure(figsize=(2, 5))
plt.pcolormesh(single_encoder_hidden_state)
plt.colorbar()

np.dot(decoder_hidden_state.T, single_encoder_hidden_state)

encoder_hidden_states = np.array([
    [1, 5, 11],
    [7, 4, 1],
    [8, 12, 2],
    [-9, 0, 1]

]).astype(float).T


def dot_product_attention_score(decoder_hidden_state, encoder_hidden_states):
    '''
    decoder_hidden_state: np.array of shape (n_features, 1)
    encoder_hidden_states: np.array of shape (n_features, n_states)

    return: np.array of shape (1, n_states)
        Array with dot product attention scores
    '''
    attention_scores = np.dot(decoder_hidden_state.T, encoder_hidden_states)
    return attention_scores


dot_product_attention_score(decoder_hidden_state, encoder_hidden_states)


def softmax(vector):
    '''
    vector: np.array of shape (n, m)

    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


weights_vector = softmax(dot_product_attention_score(decoder_hidden_state, encoder_hidden_states))

attention_vector = weights_vector.dot(encoder_hidden_states.T).T


def dot_product_attention(decoder_hidden_state, encoder_hidden_states):
    '''
    decoder_hidden_state: np.array of shape (n_features, 1)
    encoder_hidden_states: np.array of shape (n_features, n_states)

    return: np.array of shape (n_features, 1)
        Final attention vector
    '''
    softmax_vector = softmax(dot_product_attention_score(decoder_hidden_state, encoder_hidden_states))
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector


assert (attention_vector == dot_product_attention(decoder_hidden_state, encoder_hidden_states)).all()

encoder_hidden_states_complex = np.array([
    [1, 5, 11, 4, -4],
    [7, 4, 1, 2, 2],
    [8, 12, 2, 11, 5],
    [-9, 0, 1, 8, 12]

]).astype(float).T

W_mult = np.array([
    [-0.78, -0.97, -1.09, -1.79, 0.24],
    [0.04, -0.27, -0.98, -0.49, 0.52],
    [1.08, 0.91, -0.99, 2.04, -0.15]
])

# Ваша текущая задача: реализовать multiplicative attention.
# 𝑒𝑖=𝐬𝑇𝑊𝑚𝑢𝑙𝑡𝐡𝑖
# Матрица весов W_mult задана ниже. Стоит заметить, что multiplicative attention позволяет работать с состояниями энкодера и декодера различных размерностей, поэтому состояния энкодера будут обновлены. Реализуйте подсчет attention согласно формулам и реализуйте итоговую функцию `multiplicative_attention`:


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Compute attention scores
    attention_scores = np.dot(decoder_hidden_state.T, np.dot(W_mult, encoder_hidden_states))

    # Apply softmax to obtain attention weights
    attention_weights = softmax(attention_scores)

    # Compute the final attention vector
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T
    return attention_vector


result_attention_vector = multiplicative_attention(decoder_hidden_state, encoder_hidden_states_complex, W_mult)
print(result_attention_vector)

# Теперь вам предстоит реализовать additive attention.
# 𝑒𝑖=𝐯𝑇tanh(𝑊𝑎𝑑𝑑−𝑒𝑛𝑐𝐡𝑖+𝑊𝑎𝑑𝑑−𝑑𝑒𝑐𝐬)
# Матрицы весов W_add_enc и W_add_dec доступны ниже, как и вектор весов v_add. Для вычисления активации можно воспользоваться np.tanh.

v_add = np.array([[-0.35, -0.58,  0.07,  1.39, -0.79, -1.78, -0.35]]).T

W_add_enc = np.array([
    [-1.34, -0.1 , -0.38,  0.12, -0.34],
    [-1.  ,  1.28,  0.49, -0.41, -0.32],
    [-0.39, -1.38,  1.26,  1.21,  0.15],
    [-0.18,  0.04,  1.36, -1.18, -0.53],
    [-0.23,  0.96,  1.02,  0.39, -1.26],
    [-1.27,  0.89, -0.85, -0.01, -1.19],
    [ 0.46, -0.12, -0.86, -0.93, -0.4 ]
])

W_add_dec = np.array([
    [-1.62, -0.02, -0.39],
    [ 0.43,  0.61, -0.23],
    [-1.5 , -0.43, -0.91],
    [-0.14,  0.03,  0.05],
    [ 0.85,  0.51,  0.63],
    [ 0.39, -0.42,  1.34],
    [-0.47, -0.31, -1.34]
])


def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Compute attention scores
    hidden_enc = np.tanh(np.dot(W_add_enc, encoder_hidden_states) + np.dot(W_add_dec, decoder_hidden_state))
    attention_scores = np.dot(v_add.T, hidden_enc)

    # Apply softmax to obtain attention weights
    attention_weights = softmax(attention_scores)

    # Compute the final attention vector
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T
    return attention_vector

# Test the function
result_additive_attention_vector = additive_attention(decoder_hidden_state, encoder_hidden_states_complex, v_add, W_add_enc, W_add_dec)
print(result_additive_attention_vector)

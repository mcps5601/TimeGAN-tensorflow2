import tensorflow as tf
import numpy as np


def rnn_cell(module_name, hidden_dim):
    """
    Args:
        module_name: desired rnn module
        hidden_dim: dimension of hidden states
    Return:
        rnn_cell
    """
    if module_name == 'gru':
        rnn_cell = tf.keras.layers.GRUCell(
                    units=hidden_dim,
                    activation='tanh')
    if module_name == 'lstm':
        rnn_cell = tf.keras.layers.LSTMCell(
                    units=hidden_dim,
                    activation='tanh')


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.
    Args:
        batch_size: size of the random vector
        z_dim: dimension of random vector
        T_mb: time information for the random vector
        max_seq_len: maximum sequence length
    Return:
        Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i],:] = temp_Z
        Z_mb.append(temp_Z)

    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator
    Args:
        data: time-series data
        time: time information
        batch_size: the number of samples in each batch
    Return:
        X_mb: time-series data in each batch
        T_mb: time information in each batch
    """
    total_num = len(data)
    idx = np.random.permutation(total_num)
    train_idx = idx[:batch_size]
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


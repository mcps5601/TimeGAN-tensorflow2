import tensorflow as tf
import numpy as np
from .layer_norm_module import LSTMLNCell, LSTMLN
import json


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
    """Divide train and test data for both original and synthetic data.
    Args:
        data_x: original_data
        data_x_hat: generated_data
        data_t: original time
        data_t_hat: generated time
        train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    total_num = len(data)
    idx = np.random.permutation(total_num)
    train_idx = idx[:int(total_num * train_rate)]
    test_idx = idx[int(total_num * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    total_num = len(data_x_hat)
    idx = np.random.permutation(total_num)
    train_idx = idx[:int(total_num * train_rate)]
    test_idx = idx[int(total_num * train_rate):]
  
    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
    """Returns Maximum sequence length and each sequence length.
    Args:
        data: original data
    
    Returns:
        time: extracted time information
        max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
    
    return time, max_seq_len


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
    if module_name == 'lstmln':
        rnn_cell = lstmLNCell(
                    units=hidden_dim,
                    activation='tanh')
    return rnn_cell


def rnn_choices(module_name, hidden_dim):
    """
    Args:
        module_name: desired rnn module
        hidden_dim: dimension of hidden states
    Return:
        rnn_cell
    """
    if module_name == 'gru':
        rnn_model = tf.keras.layers.GRU(
                    units=hidden_dim,
                    activation='tanh',
                    return_sequences=True)
    if module_name == 'lstm':
        rnn_model = tf.keras.layers.LSTM(
                    units=hidden_dim,
                    activation='tanh',
                    return_sequences=True)
    if module_name == 'lstmln': # TODO: there may be bugs
        rnn_model = LSTMLN(
                    units=hidden_dim,
                    activation='tanh',
                    return_sequences=True)
    return rnn_model


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


def batch_generator(data, time, batch_size, use_tf_data=False):
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
    
    if use_tf_data:
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb = tf.convert_to_tensor(T_mb, dtype=tf.float32)
        X_mb = tf.data.Dataset.from_tensors(X_mb)
        T_mb = tf.data.Dataset.from_tensors(T_mb)

    return X_mb, T_mb


def MinMaxScaler(data):
    """Min_Max Normalizer.
    Args:
        data: raw_data
    Return:
        norm_data: normalized_data
        min_val: minimum values (for renormalization)
        max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def save_dict_to_json(dict_of_params, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    Return:
        a saved json file containing hyperparameters
    """

    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        dict_of_params = {k: v for k, v in dict_of_params.items()}
        json.dump(dict_of_params, f, indent=4)

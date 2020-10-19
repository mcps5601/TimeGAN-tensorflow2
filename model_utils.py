import tensorflow as tf


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


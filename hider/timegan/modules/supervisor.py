import tensorflow as tf
from .model_utils import rnn_cell, rnn_choices

class Supervisor(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.max_seq_len = args.max_seq_len

        # self.pre_rnn_cells = [rnn_cell(self.module_name, self.hidden_dim) for _ in range(self.num_layers)]
        # self.rnn_cells = tf.keras.layers.StackedRNNCells(self.pre_rnn_cells)
        # self.rnn = tf.keras.layers.RNN(self.rnn_cells,
        #                                return_sequences=True,
        #                                stateful=False,
        #                                return_state=False)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-1., input_shape=(self.max_seq_len, self.hidden_dim)),
            *[rnn_choices(self.module_name, self.hidden_dim) for _ in range(self.num_layers)]
        ])
        self.linear = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.sigmoid)

    def call(self, H, training=False):
        s_output = self.rnn(H)
        S = self.linear(s_output)
        return S
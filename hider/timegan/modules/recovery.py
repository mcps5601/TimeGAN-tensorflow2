import tensorflow as tf
from .model_utils import rnn_cell, rnn_choices

class Recovery(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers

        # self.pre_rnn_cells = [rnn_cell(args.module_name, self.hidden_dim) for _ in range(self.num_layers)]
        # self.rnn_cells = tf.keras.layers.StackedRNNCells(self.pre_rnn_cells)
        # self.rnn = tf.keras.layers.RNN(self.rnn_cells,
        #                                return_sequences=True,
        #                                stateful=False,
        #                                return_state=False)
        self.rnn = tf.keras.Sequential([
            rnn_choices(self.module_name, self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.linear = tf.keras.layers.Dense(self.feature_dim, activation=None)

    def call(self, H, training=False):
        r_output = self.rnn(H)
        X_tilde = self.linear(r_output)
        return X_tilde

import tensorflow as tf
import numpy as np
from modules.model_utils import rnn_cell

class Recovery(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.rnn_cell = rnn_cell(self.module_name, self.hidden_dim)
        self.rnn_cells = tf.keras.layers.StackedRNNCells([
            self.rnn_cell for _ in range(self.num_layers)
        ])
        self.rnn = tf.keras.layers.RNN(self.rnn_cells, return_sequences=True, 
                                    return_state=True)
        # self.rnn = tf.keras.models.Sequential([
        #     tf.keras.layers.RNN(self.rnn_cell) for _ in range(self.num_layers)
        # ])
        self.linear = tf.keras.layers.Dense(self.feature_dim, activation=tf.nn.sigmoid)

    def call(self, H, T):
        r_output, r_last_state = self.rnn(H)
        X_tilde = self.linear(r_output)
        return X_tilde
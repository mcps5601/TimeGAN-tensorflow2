import tensorflow as tf
import numpy as np
from model_utils import rnn_cell

class generator(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
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
        self.linear = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.sigmoid)

    def call(self, Z, T)
        e_output, e_last_state = self.rnn(Z)
        E = self.linear(e_output)
        return E
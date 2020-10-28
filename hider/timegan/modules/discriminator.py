import tensorflow as tf
import numpy as np
from .model_utils import rnn_cell

class Discriminator(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.rnn_cell = rnn_cell(self.module_name, self.hidden_dim)
        self.rnn_cells = tf.keras.layers.StackedRNNCells([
            self.rnn_cell for _ in range(self.num_layers)
        ])
        self.rnn = tf.keras.layers.RNN(self.rnn_cells,
                                       return_sequences=True, 
                                       stateful=False,
                                       return_state=True)
        # self.rnn = tf.keras.models.Sequential([
        #     tf.keras.layers.RNN(self.rnn_cell) for _ in range(self.num_layers)
        # ])
        self.linear = tf.keras.layers.Dense(1, activation=None)

    def call(self, H, training=False):
        d_output = self.rnn(H)
        Y_hat = self.linear(d_output[0])
        return Y_hat
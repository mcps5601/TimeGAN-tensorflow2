import tensorflow as tf
import numpy as np
from .model_utils import rnn_cell


class Generator(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        self.pre_rnn_cells = [rnn_cell(self.module_name, self.hidden_dim) for _ in range(self.num_layers)]
        self.rnn_cells = tf.keras.layers.StackedRNNCells(self.pre_rnn_cells)
        self.rnn = tf.keras.layers.RNN(self.rnn_cells, 
                                       return_sequences=True,
                                       stateful=False, 
                                       return_state=True)
        
        self.linear = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.sigmoid)

    def call(self, Z, training=False):
        e_output = self.rnn(Z)
        E = self.linear(e_output[0])
        return E
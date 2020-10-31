import tensorflow as tf


class LSTMLNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation='tanh', **kargs):
        super().__init__(**kargs)
        self.state_size = units
        self.output_size = units
        self.lstmlncell = tf.keras.layers.LSTMCell(units, activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.lstmlncell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))

        return norm_outputs, [norm_outputs]


# class LSTMLN(tf.keras.Model):
#     def __init__(self, units, activation='tanh'):
#         super().__init__()
#         self.lstmlncell = LSTMLNCell(units, activation)

#     def call()
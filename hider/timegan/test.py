import tensorflow as tf

a = tf.keras.layers.Input(shape=(10, ))
b = tf.keras.layers.Dense(10)

Layer1 = tf.keras.models.Sequential([a, b])
Layer2 = tf.keras.models.Sequential([a, b])

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = Layer1
        self.l2 = Layer2


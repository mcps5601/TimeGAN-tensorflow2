import tensorflow as tf
import numpy as np
from modules.model_utils import extract_time, random_generator, batch_generator, MinMaxScaler
from modules import Embedder, Recovery, Generator, Supervisor, Discriminator


class TimeGAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embedder = Embedder(args)
        self.recovery = Recovery(args)
        self.generator = Generator(args)
        self.supervisor = Supervisor(args)
        self.discriminator = Discriminator(args)
        self.mse_loss_fn = tf.keras.losses.mean_squared_error()

    def recovery_foward(self, X):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = self.mse_loss_fn(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

        grads = tape.gradient(E_loss0, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return E_loss_T0

    
        
    def call(self, X, Z, obj, gamma=1):






def train_timegan(ori_data, mode, args):
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)
    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    if mode == 'train':
        # Set optimizers
        embedder_optimizer = tf.keras.optimizers.Adam(epsilon=args.epsilon) #1e-8
        supervisor_optimizer = tf.keras.optimizers.Adam(epsilon=args.epsilon)
        recovery_optimizer = tf.keras.optimizers.Adam(epsilon=args.epsilon)

        print('Start Embedding Network Training')
        for itt in range(args.iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)



        for itt in range(iter)
        train_embedder()
        





if __name__ == "__main__":
    pass
import tensorflow as tf
import numpy as np
from .modules.embedder import Embedder
from .modules.recovery import Recovery
from .modules.generator import Generator
from .modules.supervisor import Supervisor
from .modules.discriminator import Discriminator
from .modules.model_utils import extract_time, random_generator, batch_generator, MinMaxScaler
import logging, os 
logging.disable(logging.WARNING) 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TimeGAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embedder = Embedder(args)
        self.recovery = Recovery(args)
        self.generator = Generator(args)
        self.supervisor = Supervisor(args)
        self.discriminator = Discriminator(args)

    def recovery_forward(self, X, optimizer):
        # initial_hidden = self.embedder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.keras.losses.mean_squared_error(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss0, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return tf.math.reduce_mean(E_loss_T0)

    def supervisor_forward(self, X, Z, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            H_hat = self.generator(Z, training=True)
            H_hat_supervise = self.supervisor(H, training=True)
            G_loss_S = tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        
        var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
        #var_list = self.generator.trainable_weights
        grads = tape.gradient(G_loss_S, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return tf.math.reduce_mean(G_loss_S)

    # def discriminator_forward(self, X, Z, obj, gamma=1):
    #     with tf.GradientTape() as tape:
    #         H = self.embedder(X, trianing=True)
    #         H_hat = self.supervisor(H, training=True)







def train_timegan(ori_data, mode, args):
    no, seq_len, dim = np.asarray(ori_data).shape

    if args.z_dim == -1:  # choose z_dim for the dimension of noises
        args.z_dim = dim

    ori_time, max_seq_len = extract_time(ori_data)
    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    if mode == 'train':
        model = TimeGAN(args)
        # Set optimizers
        E0_solver = tf.keras.optimizers.Adam(epsilon=args.epsilon)
        GS_solver = tf.keras.optimizers.Adam(epsilon=args.epsilon)

        recovery_optimizer = tf.keras.optimizers.Adam(epsilon=args.epsilon)

        # 1. Embedding network training
        print('Start Embedding Network Training')
        for itt in range(args.iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            step_e_loss = model.recovery_forward(X_mb, E0_solver)
            if itt % 1000 == 0:
                print('step: '+ str(itt) + '/' + str(args.iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))

        print('Finish Embedding Network Training')
        # 2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')
        for itt in range(args.iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)

            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            step_g_loss_s = model.supervisor_forward(X_mb, Z_mb, GS_solver)
            if itt % 1000 == 0:
                print('step: '+ str(itt)  + '/' + str(args.iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)))
        
        print('Finish Training with Supervised Loss Only')
    
    exit()

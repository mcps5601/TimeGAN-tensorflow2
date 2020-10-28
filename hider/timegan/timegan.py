import tensorflow as tf
import numpy as np
from .modules.embedder import Embedder
from .modules.recovery import Recovery
from .modules.generator import Generator
from .modules.supervisor import Supervisor
from .modules.discriminator import Discriminator
from .modules.model_utils import extract_time, random_generator, batch_generator, MinMaxScaler
import logging, os , datetime
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

    # E0_solver
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

    # GS_solver
    def supervisor_forward(self, X, Z, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            H_hat = self.generator(Z, training=True)
            H_hat_supervise = self.supervisor(H, training=True)
            G_loss_S = tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        
        var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
        grads = tape.gradient(G_loss_S, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return tf.math.reduce_mean(G_loss_S)

    # D_solver + G_solver
    def adversarial_forward(self, X, Z, optimizer, gamma=1, train_G=False, train_D=False):
        with tf.GradientTape() as tape:
            H = self.embedder(X)
            E_hat = self.generator(Z, training=True)

            # Supervisor & Recovery
            H_hat_supervise = self.supervisor(H, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            X_hat = self.recovery(H_hat)

            # Discriminator
            Y_fake = self.discriminator(H_hat, training=True)
            Y_real = self.discriminator(H, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            if train_G:
                # Generator loss
                G_loss_U = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(Y_fake), Y_fake)
                G_loss_U_e = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(Y_fake_e), Y_fake_e)
                G_loss_S = tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                G_loss_V1 = tf.math.reduce_mean(tf.math.abs(tf.math.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6)
                                                - tf.math.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
                G_loss_V2 = tf.math.reduce_mean(tf.math.abs((tf.nn.moments(X_hat,[0])[0])
                                                - (tf.nn.moments(X,[0])[0])))
                G_loss_V = G_loss_V1 + G_loss_V2
                ## Sum of all G_losses
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.math.sqrt(G_loss_S) + 100 * G_loss_V

            elif not train_G:
                # Discriminator loss
                D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(Y_real), Y_real)
                D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(Y_fake), Y_fake)
                D_loss_fake_e = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(Y_fake_e), Y_fake_e)
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if train_G:
            GS_var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
            GS_grads = tape.gradient(G_loss, GS_var_list)
            optimizer.apply_gradients(zip(GS_grads, GS_var_list))
            
            return G_loss_U, G_loss_S, G_loss_V

        elif train_D:
            D_var_list = self.discriminator.trainable_weights
            D_grads = tape.gradient(D_loss, D_var_list)
            optimizer.apply_gradients(zip(grads, D_var_list))

            return tf.math.reduce_mean(D_loss)

        elif not train_D:
            print("Checking if D_loss > 0.15")
            return tf.math.reduce_mean(D_loss)

    # E_solver
    def embedding_forward_joint(self, X, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.keras.losses.mean_squared_error(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

            H_hat_supervise = self.supervisor(H)
            G_loss_S = tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
         
            E_loss = E_loss0 + 0.1 * G_loss_S
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return tf.math.reduce_mean(E_loss)     


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
        G_solver = tf.keras.optimizers.Adam(epsilon=args.epsilon)
        E_solver = tf.keras.optimizers.Adam(epsilon=args.epsilon)
        
        print('Set up Tensorboard')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('tensorboard', current_time + '-' + args.exp_name)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # 1. Embedding network training
        print('Start Embedding Network Training')
        for itt in range(args.iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            step_e_loss = model.recovery_forward(X_mb, E0_solver)
            if itt % 100 == 0:
                print('step: '+ str(itt) + '/' + str(args.iterations) +
                      ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))
                # Write to Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('Embedding_loss', np.round(np.sqrt(step_e_loss),4), step=itt)

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

        # 3. Joint Training
        print('Start Joint Training')
        for itt in range(args.iterations):
            # Generator training (two times as discriminator training)
            for g_more in range(2):
                X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
                Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)
                step_g_loss_u, step_g_loss_s, step_g_loss_v = adversarial_forward(X_mb, Z_mb,
                                                                                  G_solver,
                                                                                  train_G=True,
                                                                                  train_D=False)
                step_e_loss_t0 = embedding_forward_joint(X_mb, E_solver)

            # Discriminator training
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)
            check_d_loss = adversarial_forward(X_mb, Z_mb, G_solver, train_G=False, train_D=True)


    
    exit()

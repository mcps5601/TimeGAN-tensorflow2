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


class TimeGAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embedder = Embedder(args)
        self.recovery = Recovery(args)
        self.generator = Generator(args)
        self.supervisor = Supervisor(args)
        self.discriminator = Discriminator(args)
        self.mse = tf.keras.losses.MeanSquaredError()

    # E0_solver
    def recovery_forward(self, X, optimizer):
        # initial_hidden = self.embedder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = self.mse(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss0, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss_T0

    # GS_solver
    def supervisor_forward(self, X, Z, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            H_hat = self.generator(Z, training=True)
            H_hat_supervise = self.supervisor(H, training=True)
            G_loss_S = self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        
        var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
        grads = tape.gradient(G_loss_S, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return G_loss_S

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
                G_loss_U = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake), Y_fake, from_logits=True)
                )
                G_loss_U_e = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True)
                )
                G_loss_S = tf.math.reduce_mean(
                    tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                )
                # Difference in "variance" between X_hat and X
                G_loss_V1 = tf.math.reduce_mean(tf.math.abs(tf.math.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
                                                - tf.math.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
                # Difference in "mean" between X_hat and X
                G_loss_V2 = tf.math.reduce_mean(tf.math.abs((tf.nn.moments(X_hat, [0])[0])
                                                - (tf.nn.moments(X, [0])[0])))
                G_loss_V = G_loss_V1 + G_loss_V2
                #G_loss_V = tf.math.add(G_loss_V1, G_loss_V2)
                ## Sum of all G_losses
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.math.sqrt(G_loss_S) + 100 * G_loss_V

            elif not train_G:
                # Discriminator loss
                D_loss_real = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_real), Y_real, from_logits=True)
                )
                D_loss_fake = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake), Y_fake, from_logits=True)
                )
                D_loss_fake_e = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True)
                )
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if train_G:
            GS_var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
            GS_grads = tape.gradient(G_loss, GS_var_list)
            optimizer.apply_gradients(zip(GS_grads, GS_var_list))
            
            return G_loss_U, G_loss_S, G_loss_V

        elif train_D:
            D_var_list = self.discriminator.trainable_weights
            D_grads = tape.gradient(D_loss, D_var_list)
            optimizer.apply_gradients(zip(D_grads, D_var_list))

            return D_loss

        elif not train_D:
            # Checking if D_loss > 0.15
            return D_loss

    # E_solver
    def embedding_forward_joint(self, X, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(X, X_tilde)
            )
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

            H_hat_supervise = self.supervisor(H)
            G_loss_S = tf.math.reduce_mean(
                self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            )
            E_loss = E_loss0 + 0.1 * G_loss_S
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss

    # Inference
    def generate(self, Z, ori_data_num, max_val, min_val):
        """
        Args:
            Z: input random noises
            ori_data_num: the first dimension of ori_data.shape
            max_val: the maximum value of MinMaxScaler(ori_data)
            min_val: the minimum value of MinMaxScaler(ori_data)
        Return:
            generated_data: synthetic time-series data
        """
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        generated_data_curr = self.recovery(H_hat)
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, :ori_time[i], :]
            generated_data.append(temp)
        
        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val
        
        return generated_data

    # Direct inference (testing)
    def generator_inference(self, z_dim, ori_data, model_dir):
        """
        Args:
            Z: input random noises
            ori_data: the original dataset (for information extraction)
            trained_model: trained Generator
        Return:
            generated_data: synthetic time-series data
        """
        no, seq_len, dim = np.asarray(ori_data).shape
        ori_time, max_seq_len = extract_time(ori_data)
        # Normalization
        _, min_val, max_val = MinMaxScaler(ori_data)


        if z_dim == -1:  # choose z_dim for the dimension of noises
            z_dim = dim
        Z = random_generator(no, z_dim, ori_time, max_seq_len)

        # Load models
        self.recovery.load_weights(model_dir)
        self.supervisor.load_weights(model_dir)
        self.generator.load_weights(model_dir)

        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        generated_data_curr = self.recovery(H_hat)
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, :ori_time[i], :]
            generated_data.append(temp)
        
        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val
        
        return generated_data


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
        D_solver = tf.keras.optimizers.Adam(epsilon=args.epsilon)
        
        print('Set up Tensorboard')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('tensorboard', current_time + '-' + args.exp_name)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # 1. Embedding network training
        print('Start Embedding Network Training')
        for itt in range(1):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            step_e_loss = model.recovery_forward(X_mb, E0_solver)
            if itt % 10 == 0:
                print('step: '+ str(itt) + '/' + str(args.iterations) +
                      ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))
                # Write to Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('Embedding_loss', np.round(np.sqrt(step_e_loss),4), step=itt)

        print('Finish Embedding Network Training')

        # 2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')
        for itt in range(1):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)

            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            step_g_loss_s = model.supervisor_forward(X_mb, Z_mb, GS_solver)
            if itt % 10 == 0:
                print('step: '+ str(itt)  + '/' + str(args.iterations) +', s_loss: '
                              + str(np.round(np.sqrt(step_g_loss_s),4)))
                # Write to Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('Supervised_loss', np.round(np.sqrt(step_g_loss_s),4), step=itt)
        
        print('Finish Training with Supervised Loss Only')

        # 3. Joint Training
        print('Start Joint Training')
        for itt in range(5):
            # Generator training (two times as discriminator training)
            for g_more in range(2):
                X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
                Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)
                
                X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
                Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

                step_g_loss_u, step_g_loss_s, step_g_loss_v = model.adversarial_forward(X_mb, Z_mb,
                                                                                  G_solver,
                                                                                  train_G=True,
                                                                                  train_D=False)
                step_e_loss_t0 = model.embedding_forward_joint(X_mb, E_solver)

            # Discriminator training
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
            Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)

            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            check_d_loss = model.adversarial_forward(X_mb, Z_mb, D_solver, train_G=False, train_D=False)
            if (check_d_loss > 0.15): 
                step_d_loss = model.adversarial_forward(X_mb, Z_mb, D_solver, train_G=False, train_D=True)
            else:
                step_d_loss = check_d_loss

            if itt % 100 ==0:
                print('step: '+ str(itt) + '/' + str(args.iterations) + 
                      ', d_loss: ' + str(np.round(step_d_loss, 4)) + 
                      ', g_loss_u: ' + str(np.round(step_g_loss_u, 4)) + 
                      ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) + 
                      ', g_loss_v: ' + str(np.round(step_g_loss_v, 4)) + 
                      ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0), 4)))
                # Write to Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('Joint/Discriminator',
                                      np.round(step_d_loss, 4), step=itt)
                    tf.summary.scalar('Joint/Generator',
                                      np.round(step_g_loss_u, 4), step=itt)
                    tf.summary.scalar('Joint/Supervisor',
                                      np.round(step_g_loss_s, 4), step=itt)
                    tf.summary.scalar('Joint/Moments',
                                      np.round(step_g_loss_v, 4), step=itt)
                    tf.summary.scalar('Joint/Embedding',
                                      np.round(step_e_loss_t0, 4), step=itt)        
        print('Finish Joint Training')
    
        ## Synthetic data generation
        Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)
        generated_data = generate(Z_mb, no, max_val, min_val)
        
        return generated_data
import tensorflow as tf
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
        
    def call(self, ori_data):
        ori_time, max_seq_len = extract_time(ori_data)

        # Normalization
        ori_data, min_val, max_val = MinMaxScaler(ori_data)
        






if __name__ == "__main__":
    pass
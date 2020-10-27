##### TensorFlow2 #####
import tensorflow as tf
import os, random
import numpy as np

# This function can be successfully used in TensorFlow 2.x and Keras for reproducibility.
def tf2_set_seed(seed):
    """
    Args:
        seed: an integer number to initialize a pseudorandom number generator
    """
    tf.random.set_seed(seed)
    #tf.compat.v1.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

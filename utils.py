import os, sys, random, logging, json, multiprocessing
import traceback
import contextlib
import numpy as np

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# @contextlib.contextmanager
# def temp_seed_numpy(seed):
#     """Set a temporary numpy seed: set the seed at the beginning of this context, then at the end, restore random 
#     state to what it was before.

#     Args:
#         seed (int): Random seed to use.
#     """
#     state = np.random.get_state()
#     np.random.seed(seed)
#     try:
#         yield
#     finally:
#         np.random.set_state(state)


def tf_set_log_level(level):
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.getLogger("tensorflow").setLevel(level)


@contextlib.contextmanager
def in_progress(stage):
    print(f"{stage}...")
    try:
        yield
    finally:
        print(f"{stage} DONE")


# @contextlib.contextmanager
# def tf_fixed_seed_seesion(seed):
#     fix_all_random_seeds(seed)
#     try:
#         yield
#     finally:
#         tf.compat.v1.reset_default_graph()


def read_competition_config(config_path):
    config_path = os.path.realpath(config_path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = dict()
    return config


def redact_exception(ex: Exception) -> str:
    """Log an exception in a redacted way. Logs the traceback code line and function name but not the message.

    Args:
        ex (Exception): The exception to log.

    Returns:
        str: The redacted exception message.
    """
    msg = ""
    name = ex.__class__.__name__ if hasattr(ex, "__class__") else "Exception"
    _, _, e_traceback = sys.exc_info()
    msg += "Traceback (most recent call last):\n"
    for filename, linenum, funname, line in traceback.extract_tb(e_traceback):
        msg += f'  File "{filename}", line {linenum}, in {funname}\n    {line}\n'
    msg += f"{name}: [MESSAGE REDACTED]"
    return msg


@contextlib.contextmanager
def dump_redacted_exc(dump_path: str):
    """A context manager that wraps code making sure that exceptions are logged to `dump_path` in a redacted manner.

    Args:
        dump_path (Exception): The filepath to dump redacted exceptions to.
    
    Raises:
        ex: Exception logged compliantly.
    """
    try:
        if os.path.exists(dump_path):
            os.remove(dump_path)
        yield
    except Exception as ex:
        msg = redact_exception(ex)
        with open(dump_path, "w") as f:
            f.writelines(msg)
        raise


def time_limited(func, time_s, exc_dump_path, *args, **kwargs):
    p = multiprocessing.Process(target=func, name="Foo", args=args, kwargs=kwargs)
    p.start()
    p.join(time_s)
    if p.is_alive():
        msg = "Solution terminated due to running in excess of time limit."
        print(msg)
        p.terminate()
        p.join()
        # Dump a mock an exception:
        if os.path.exists(exc_dump_path):
            os.remove(exc_dump_path)
        with open(exc_dump_path, "w") as f:
            f.writelines("RuntimeError: " + msg)
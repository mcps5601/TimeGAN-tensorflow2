"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Modified by Ying-Jia Lin
Contact: jsyoon0823@gmail.com

-----------------------------

Note: We use TimeGAN or noise addition as a hider and KNN as the seeker as examples.

Pipeline
Step 1: Load and preprocess dataset
Step 2: Run hider algorithm
Step 3: Define enlarge data and its label
Step 4: Run seeker algorithm
Step 5: Evaluation
  - feature-prediction
  - one-step-ahead-prediction
  - reidentification-score
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, joblib, time, pickle, shutil
import argparse
import numpy as np
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings("ignore")

from utils import tf2_set_seed, in_progress, tf_set_log_level
from hider.timegan.timegan import train_timegan
from hider.timegan import dp_timegan
# from hider.add_noise import add_noise
from seeker.knn_seeker import knn_seeker
from seeker.binary_predictor import binary_predictor
from data.data_utils import data_division
from data.data_preprocess import data_preprocess
from metrics.metric_utils import feature_prediction, one_step_ahead_prediction, reidentify_score
from sklearn.metrics import confusion_matrix

def main(args):
    """Hide-and-Seek Privacy Challenge main function.

    Args:
    - data_name: amsterdam or stock
    - max_seq_len: maximum sequence length
    - train_rate: ratio of training data
    - feature_prediction_no: the number of features to be predicted for evaluation
    - seed: random seed for train / test data division
    - hider_model: timegan or add_noise
    - noise_size: size of the noise for add_noise hider

    Returns:
    - feat_pred: feature prediction results (original & new)
    - step_ahead_pred: step ahead prediction results (original & new)
    - reidentification_score: reidentification score between hider and seeker
    """
    # Set random seeds
    tf2_set_seed(args.seed)

    ## Load & preprocess data
    if args.data_name == 'amsterdam':
        if args.use_gain:
            print("Using GAIN-imputed data")
            file_name = '/data/nips_hns/hide-and-seek/data/GAIN/amsterdam.csv'
            ori_data = data_preprocess(file_name, args.max_seq_len)
        else:
            if os.path.exists('data/amsterdam/amsterdam-bin.jlb'):
                ori_data = joblib.load('data/amsterdam/amsterdam-bin.jlb')
            else:
                file_name = 'data/amsterdam/train_longitudinal_data.csv'
                ori_data = data_preprocess(file_name, args.max_seq_len)

    elif args.data_name == 'stock':
        with open('data/public_data/public_' + args.data_name + '_data.txt', 'rb') as fp:
            ori_data = pickle.load(fp)
            ori_data = np.asarray(ori_data)

    # Divide the data into training and testing
    divided_data, _ = data_division(ori_data, seed = args.seed, divide_rates = [args.train_rate, 1-args.train_rate])
    args.feature_dim = ori_data.shape[-1]

    train_data = np.asarray(divided_data[0])
    test_data = np.asarray(divided_data[1])

    print('Finish data loading: ' + str(args.data_name))

    # For CuDNN bug
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    # for gpu in gpus:
    #     tf.config.experimental.set_visible_devices(gpu, device_type='GPU')
    #     tf.config.experimental.set_memory_growth(gpu, True)

    ## Run hider algorithm
    hider_start = time.time()
    if args.hider_model == 'timegan':
        if args.use_dpsgd:
            generated_data, train_log_dir = dp_timegan.train_timegan(train_data, 'train', args)
        else:
            generated_data, train_log_dir = train_timegan(train_data, 'train', args)
    elif args.hider_model == 'add_noise':
        generated_data = add_noise.add_noise(train_data, args.noise_size)

    print('Finish hider algorithm (' + args.hider_model  + ') training')
    hider_end = time.time()

    # Save the train and generated data for visualization
    with open(os.path.join(train_log_dir, "ori.pickle"), "wb") as fb:
        pickle.dump(train_data, fb)
    with open(os.path.join(train_log_dir, "test.pickle"), "wb") as fb:
        pickle.dump(test_data, fb)
    with open(os.path.join(train_log_dir, "new.pickle"), "wb") as fb:
        pickle.dump(generated_data, fb)

    ## Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis = 0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0],]), np.zeros([test_data.shape[0],])), axis = 0)

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    ## Run seeker algorithm
    seeker_start = time.time()
    if args.seeker_model == 'binary_predictor':
        reidentified_data = binary_predictor(generated_data, enlarge_data, train_log_dir)
    elif args.seeker_model == 'knn':
        reidentified_data = knn_seeker(generated_data, enlarge_data)

    print('Finish seeker algorithm (' + args.seeker_model  + ') training')
    seeker_end = time.time()

    print('Hider needs {} sec'.format(hider_end - hider_start))
    print('Seeker needs {} sec'.format(seeker_end - seeker_start))

    ## Evaluate the performance
    # 1. Feature prediction
    feat_idx = np.random.permutation(train_data.shape[2])[:args.feature_prediction_no]
    #feat_idx = [69, 70]
    ori_feat_pred_perf = feature_prediction(train_data, test_data, feat_idx)
    new_feat_pred_perf = feature_prediction(generated_data, test_data, feat_idx)

    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    print('Feature prediction results: ' +
            '(1) Ori: ' + str(np.round(ori_feat_pred_perf, 4)) +
            '(2) New: ' + str(np.round(new_feat_pred_perf, 4)))

    # 2. One step ahead prediction
    ori_step_ahead_pred_perf = one_step_ahead_prediction(train_data, test_data)
    new_step_ahead_pred_perf = one_step_ahead_prediction(generated_data, test_data)

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print('One step ahead prediction results: ' + 
            '(1) Ori: ' + str(np.round(ori_step_ahead_pred_perf, 4)) + 
            '(2) New: ' + str(np.round(new_step_ahead_pred_perf, 4)))

    # 3. Reidentification score
    reidentification_score = reidentify_score(enlarge_data_label, reidentified_data)
    confusion_matrix(enlarge_data_label, reidentified_data)
    print('Reidentification score: ' + str(np.round(reidentification_score, 4)))

    return feat_pred, step_ahead_pred, reidentification_score

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

###
if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        default='ae_zero_imp_10',
        type=str)
    parser.add_argument(
        '--data_name',
        choices=['amsterdam','stock'],
        default='amsterdam',
        type=str)
    parser.add_argument(
        '--use_gain',
        default=False,
        type=str2bool)
    parser.add_argument(
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)
    parser.add_argument(
        '--feature_prediction_no',
        default=10,
        type=int)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--hider_model',
        choices=['timegan','add_noise'],
        default='timegan',
        type=str)
    parser.add_argument(
        '--noise_size',
        default=0.1,
        type=float)
    parser.add_argument(
        '--seeker_model',
        choices=['binary_predictor','knn'],
        default='binary_predictor',
        type=str)
    ##### hider params #####
    parser.add_argument(
        '--gen_type',
        choices=['gan', 'autoencoder'],
        default='autoencoder',
        type=str)
    parser.add_argument(
        '--module_name',
        choices=['gru','lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--epsilon',
        default=1e-8,
        type=float)
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str)
    parser.add_argument(
        '--use_dpsgd',
        default=False,
        type=str2bool)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--z_dim',
        default=-1,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=10,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--embedding_iterations',
        default=2000,
        type=int)
    parser.add_argument(
        '--supervised_iterations',
        default=500,
        type=int)
    parser.add_argument(
        '--joint_iterations',
        default=6000,
        type=int)
    parser.add_argument(
        '--eta',
        default=0.1,
        type=int)
    ##### DP params #####
    parser.add_argument(
        '--l2_norm_clip',
        default=1.0,
        type=float)
    parser.add_argument(
        '--noise_multiplier',
        default=0.1,
        type=float)
    parser.add_argument(
        '--dp_lr',
        default=0.15,
        type=float)

    args = parser.parse_args()

    # Call main function
    feat_pred, step_ahead_pred, reidentification_score = main(args)

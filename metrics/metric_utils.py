"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

metric_utils.py

(1) reidentify_score: Return the reidentification score.
(2) feature_prediction: use the other features to predict a certain feature
(3) one_step_ahead_prediction: use the previous time-series to predict one-step ahead feature values
"""

# Necessary packages
import numpy as np
from metrics.general_rnn import GeneralRNN
from sklearn.metrics import accuracy_score, roc_auc_score

def reidentify_score(enlarge_label, pred_label):
  """Return the reidentification score.
  
  Args:
    - enlarge_label: 1 for train data, 0 for other data
    - pred_label: 1 for reidentified data, 0 for not reidentified data
    
  Returns:
    - accuracy: reidentification score
  """  
  accuracy = accuracy_score(enlarge_label, pred_label > 0.5)  
  return accuracy


def rmse_error (y_true, y_pred):
  """User defined root mean squared error.
  
  Args:
    - y_true: true labels
    - y_pred: predictions
    
  Returns:
    - computed_rmse: computed rmse loss
  """
  # Exclude masked labels
  idx = (y_true >= 0) * 1
  # Mean squared loss excluding masked labels
  computed_mse = np.sum(idx * ((y_true - y_pred)**2)) / np.sum(idx)
  computed_rmse = np.sqrt(computed_mse)
  return computed_rmse


def feature_prediction (train_data, test_data, index):
  """Use the other features to predict a certain feature.
  
  Args:
    - train_data: training time-series
    - test_data: testing time-series
    - index: feature index to be predicted
    
  Returns:
    - perf: average performance of feature predictions (in terms of AUC or MSE)
  """
  
  # Parameters
  no, seq_len, dim = train_data.shape
  
  # Set model parameters
  model_parameters = {'task': 'regression',
                      'model_type': 'gru',
                      'h_dim': dim,
                      'n_layer': 3,
                      'batch_size': 128,
                      'epoch': 20,
                      'learning_rate': 0.001}
  
  # Output initialization
  perf = list()
  
  # For each index
  for idx in index:
    # Set training features and labels
    train_x = np.concatenate((train_data[:, :, :idx], train_data[:, :, (idx+1):]), axis= 2 )
    train_y = np.reshape(train_data[:, :, idx], [no, seq_len, 1])
    
    # Set testing features and labels
    test_x = np.concatenate((test_data[:, :, :idx], test_data[:, :, (idx+1):]), axis= 2 )
    test_y = np.reshape(test_data[:, :, idx], [test_data.shape[0], seq_len, 1])
    
    # Train the predictive model
    if len(np.unique(train_y)) == 2: model_parameters['task'] = 'classification'
    general_rnn = GeneralRNN(model_parameters)    
    general_rnn.fit(train_x, train_y)
    test_y_hat = general_rnn.predict(test_x)
    
    # Evaluate the trained model
    test_y = np.reshape(test_y, [-1])
    test_y_hat = np.reshape(test_y_hat, [-1])
    
    if model_parameters['task'] == 'classification':
      temp_perf = roc_auc_score(test_y, test_y_hat)
    elif model_parameters['task'] == 'regression':
      temp_perf = rmse_error(test_y, test_y_hat)
      
    perf.append(temp_perf)
    
  return perf
      
      
def one_step_ahead_prediction (train_data, test_data):
  """Use the previous time-series to predict one-step ahead feature values.
  
  Args:
    - train_data: training time-series
    - test_data: testing time-series
    
  Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
  """
  
  # Parameters
  no, seq_len, dim = train_data.shape
  
  # Set model parameters
  model_parameters = {'task': 'regression',
                      'model_type': 'gru',
                      'h_dim': dim,
                      'n_layer': 3,
                      'batch_size': 128,
                      'epoch': 20,
                      'learning_rate': 0.001}
  
  # Set training features and labels
  train_x = train_data[:, :-1, :]
  train_y = train_data[:, 1:, :]
  
  # Set testing features and labels
  test_x = test_data[:, :-1, :]
  test_y = test_data[:, 1:, :]
    
  # Train the predictive model
  if len(np.unique(train_y)) == 2: model_parameters['task'] = 'classification'
    
  general_rnn = GeneralRNN(model_parameters)    
  general_rnn.fit(train_x, train_y)
  test_y_hat = general_rnn.predict(test_x)
    
  # Evaluate the trained model
  test_y = np.reshape(test_y, [-1])
  test_y_hat = np.reshape(test_y_hat, [-1])
    
  if model_parameters['task'] == 'classification':
    perf = roc_auc_score(test_y, test_y_hat)
  elif model_parameters['task'] == 'regression':
    perf = rmse_error(test_y, test_y_hat)
    
  return perf
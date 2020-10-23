"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

data_utils.py

(1) MinMaxScaler: Min Max normalizer
(2) data_division: Divide the dataset into sub datasets.
(3) subset_sampling: Sample the original data to construct multiple sub data
"""

## Necessary Packages
import numpy as np
import random


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def data_division (data, seed, divide_rates):
  """Divide the dataset into sub datasets.
  
  Args:
    - data: original data (list format)
    - seed: random seed
    - divide_rates: ratio for each division
    
  Returns:
    - divided_data: divided data (list format)
    - divided_index: divided data index (list format)
  """
  # sum of the division rates should be 1
  assert sum(divide_rates) == 1
  
  # Output initialization
  divided_data = list()
  divided_index = list()
  
  # Set index
  no = len(data)
  random.seed(seed)
  index = np.random.permutation(no)

  # Set divided index & data
  for i in range(len(divide_rates)):
    temp_idx = index[int(no*sum(divide_rates[:i])):int(no*sum(divide_rates[:(i+1)]))]
    divided_index.append(temp_idx)
    
    temp_data = [data[j] for j in temp_idx]
    divided_data.append(temp_data)
  
  return divided_data, divided_index


def subset_sampling (data, seed, subset_rates):
  """Sample the original data to construct multiple sub data.
  
  Args:
    - data: original data (list format)
    - seed: random seed
    - subset_rates: ratio for each division
    
  Returns:
    - divided_data: divided data (list format)
    - divided_index: divided data index (list format)
  """
  # Output initialization
  divided_data = list()
  divided_index = list()
  
  # Set index
  no = len(data)

  # Set divided index & data
  for i in range(len(subset_rates)):
    random.seed(seed+i)
    
    index = np.random.permutation(no)[int(no*subset_rates[i])]
    divided_index.append(index)
    
    temp_data = [data[j] for j in index]
    divided_data.append(temp_data)
  
  return divided_data, divided_index  
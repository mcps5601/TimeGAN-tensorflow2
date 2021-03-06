"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Modified by Ying-Jia Lin

-----------------------------

(1) data_preprocess: Load the data and preprocess for 3d numpy array
(2) imputation: Impute missing data using bfill, ffill and median imputation
"""

## Necessary packages
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def data_preprocess(file_name, max_seq_len, imp_method):
    """Load the data and preprocess for 3d numpy array.

    Args:
    - file_name: CSV file name
    - max_seq_len: maximum sequence length

    Returns:
    - processed_data: preprocessed data
    """

    # Load data  
    ori_data = pd.read_csv(file_name)

    # Parameters
    uniq_id = np.unique(ori_data['admissionid'])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 1

    if imp_method == 'mode':
        # Get mode of each columns
        keys = ori_data.columns[2:]
        vals = []
        for key in keys:
            val = ori_data[str(key)].dropna().to_numpy()
            vals.append(stats.mode(val).mode[0])

    elif imp_method == 'median':
        vals = ori_data.median()

    # Preprocessing
    scaler = MinMaxScaler()
    scaler.fit(ori_data)

    # Output initialization
    processed_data = -np.ones([no, max_seq_len, dim])
    time = []

    # For each uniq id
    for i in tqdm(range(no)):
        # Extract the time-series data with a certain admissionid
        idx = ori_data.index[ori_data['admissionid'] == uniq_id[i]]
        curr_data = ori_data.iloc[idx]

        # Preprocess time
        curr_data['time'] = curr_data['time'] - np.min(curr_data['time'])    

        # Impute missing data
        curr_data = imputation(curr_data, vals)

        # MinMax Scaling    
        curr_data = scaler.transform(curr_data)

        # Assign to the preprocessed data (Excluding ID)        
        curr_no = len(curr_data)
        if curr_no >= max_seq_len:
            processed_data[i, :, :] = curr_data[:max_seq_len, 1:]
            time.append(max_seq_len)
        else:
            processed_data[i, -curr_no:, :] = (curr_data)[:, 1:]
            time.append(curr_no)

    return processed_data, time


def imputation(curr_data, vals):
    """Impute missing data using bfill, ffill and median imputation.

    Args:
    - curr_data: current pandas dataframe
    - median_vals: median values for each column

    Returns:
    - imputed_data: imputed pandas dataframe
    """
    imputed_data = curr_data.fillna(vals)

    return imputed_data

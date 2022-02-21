# %%

# Generic Imports

import pandas as pd
import numpy as np
import json
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score

# LR Imports

from lr import predict

# PCA imports

from gamma_pca import (
    get_gamma_params,
    compute_corr_matrix,
    compute_gamma_matrix,
)

# %%

x_columns = [
    'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
    'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
    'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30'
]


def get_x_y(cancer_df):

    X = cancer_df[x_columns]
    y = cancer_df['g']

    # cancer_df.describe()
    return X, y


def load_model_params(model_path):
    with open(model_path, "r") as f:
        model_params = json.load(f)
    return model_params['theta'], model_params['n_components']


# %%


def main():
    gamma_matrix = np.zeros((len(x_columns) + 2, len(x_columns) + 2))
    chunksize = 10**3
    x_arr_chunks = []
    y_arr_chunks = []
    for chunk in pd.read_csv("cancer_1M.csv", chunksize=chunksize):
        x, y = get_x_y(chunk)
        x_arr = x.values.T
        y_arr = y.values
        gamma_matrix += compute_gamma_matrix(x_arr, y_arr)
        x_arr_chunks.append(x_arr)
        y_arr_chunks.append(y_arr)

    x_arr_chunks[0].shape
    x_arr = np.concatenate(x_arr_chunks, axis=1)
    y_arr = np.concatenate(y_arr_chunks, axis=0)

    theta, n_components = load_model_params('model_params.json')
    theta = np.array(object=theta)
    # Get gamma matrix params
    n, L, Q = get_gamma_params(gamma_matrix)

    # Compute correlation matrix
    corr_matrix = compute_corr_matrix(gamma_matrix, len(x_columns))

    # Generate Eigen vectors and values
    u, s, vh = np.linalg.svd(corr_matrix, full_matrices=True)

    # Dimensionality reduction
    x_arr = x_arr.T @ np.diag(s)[:, :n_components]

    print(f"x_arr.shape: {x_arr.shape}")

    # Load Standard Scaler
    with open('standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    # Normalize
    x_arr = standard_scaler.fit_transform(x_arr)

    # Predict
    y_pred = predict(x_arr, theta)

    # Confusion matrix
    print(confusion_matrix(y_arr, y_pred))

    # Accuracy
    print(accuracy_score(y_arr, y_pred))


# %%

if __name__ == '__main__':
    main()

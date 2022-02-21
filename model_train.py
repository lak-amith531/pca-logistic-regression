# %%

import pandas as pd
import numpy as np
import json
import pickle

from sklearn.preprocessing import StandardScaler

# LR Imports

from lr import logistic_regression

# PCA imports

from gamma_pca import (
    get_gamma_params,
    compute_corr_matrix,
    compute_gamma_matrix,
)

# %%
x_columns = [
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'x9',
    'x10',
    'x11',
    'x12',
    'x13',
    'x14',
    'x15',
    'x16',
    'x17',
    'x18',
    'x19',
    'x20',
    'x21',
    'x22',
    'x23',
    'x24',
    'x25',
    'x26',
    'x27',
    'x28',
    'x29',
    'x30',
]


def get_x_y(cancer_df):

    X = cancer_df[x_columns]
    y = cancer_df['g']

    # cancer_df.describe()
    return X, y


# %%


def main():

    # Chunkwise
    gamma_matrix = np.zeros((len(x_columns) + 2, len(x_columns) + 2))
    chunksize = 10**3
    x_arr_chunks = []
    y_arr_chunks = []
    for chunk in pd.read_csv("cancer_100k.csv", chunksize=chunksize):
        x, y = get_x_y(chunk)
        x_arr = x.values.T
        y_arr = y.values
        gamma_matrix += compute_gamma_matrix(x_arr, y_arr)
        x_arr_chunks.append(x_arr)
        y_arr_chunks.append(y_arr)

    # Concatenate chunks into one array
    x_arr = np.concatenate(x_arr_chunks, axis=1)
    y_arr = np.concatenate(y_arr_chunks, axis=0)

    # Get gamma matrix params
    n, L, Q = get_gamma_params(gamma_matrix)

    # Compute correlation matrix
    corr_matrix = compute_corr_matrix(gamma_matrix, len(x_columns))

    # Generate Eigen vectors and values
    u, s, vh = np.linalg.svd(corr_matrix, full_matrices=True)

    # Only consider eigen values greater than 1
    n_components = len(s[s > 1])
    x_arr = x_arr.T @ np.diag(s)[:, :n_components]

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_arr,
    #     y_arr,
    #     test_size=0.2,
    #     random_state=0,
    # )

    # standard_scaler = StandardScaler()
    # x_train = standard_scaler.fit_transform(x_train)
    # x_test = standard_scaler.transform(x_test)

    # Using entire dataset to build model
    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_arr)
    y_train = y_arr

    theta = logistic_regression(x_train, y_train, 1, 1000)

    # Save Theta values and n_components value
    with open('model_params.json', 'w') as fp:
        json.dump(
            {
                'theta': theta.tolist(),
                'n_components': n_components,
            },
            fp,
            indent=2,
        )

    # Save the scaler
    with open('standard_scaler.pkl', 'wb') as fp:
        pickle.dump(standard_scaler, fp)


# %%

if __name__ == "__main__":
    main()

# %%

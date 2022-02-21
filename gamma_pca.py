import numpy as np


def compute_gamma_matrix(x_arr, y_arr):

    x_arr.shape
    x_arr = np.r_[[np.ones(
        x_arr.shape[1],
        np.int32,
    )], x_arr]

    x_arr = np.r_[x_arr, [y_arr]]
    return np.dot(x_arr, x_arr.T)


def get_gamma_params(gamma_matrix):
    num_rows = gamma_matrix.shape[0]
    num_cols = gamma_matrix.shape[1]
    n = gamma_matrix[0][0]
    L = gamma_matrix[1:num_rows - 1, 0]
    Q = gamma_matrix[1:num_rows - 1, 1:num_cols - 1]
    return n, L, Q


def compute_corr_matrix(gamma_matrix, no_of_dims):
    corr_matrix = np.zeros((no_of_dims, no_of_dims))
    n, L, Q = get_gamma_params(gamma_matrix)
    for i in range(no_of_dims):
        for j in range(no_of_dims):
            corr_matrix[i][j] = (n * Q[i][j] -
                                 (L[i] * L[j])) / (np.sqrt(n * Q[i][i] -
                                                           (L[i] * L[i])) *
                                                   np.sqrt(n * Q[j][j] -
                                                           (L[j] * L[j])))

    return corr_matrix

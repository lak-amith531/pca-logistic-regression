import numpy as np
import math


def generate_vector(x):
    """
    Taking the original independent variables matrix and
    add a row of 1 which corresponds to x_0
        Parameters:
        X:  independent variables matrix
        Return value: the matrix that contains all the
        values in the dataset, not include the outcomes variables.
    """
    vector_x = np.c_[np.ones((len(x), 1)), x]
    return vector_x


def theta_init(x):
    """
    Generate an initial value of vector θ from the original
    independent variables matrix
        Parameters:
        X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(x[0]) + 1, 1)
    return theta


def sigmoid_function(x):
    """
    Calculate the sigmoid value of the inputs
        Parameters:
            X:  values
        Return value: the sigmoid value
    """
    return 1 / (1 + math.e**(-x))


def logistic_regression(x, y, learningrate, iterations):
    """
    Find the Logistics regression model for the data set
        Parameters:
            X: independent variables matrix
            y: dependent variables matrix
            learningrate: learningrate of Gradient Descent
            iterations: the number of iterations
        Return value: the final theta vector
    """
    y_new = np.reshape(y, (len(y), 1))
    cost_lst = []
    vector_x = generate_vector(x)
    theta = theta_init(x)
    m = len(x)
    for _ in range(iterations):
        gradients = 2 / m * vector_x.T.dot(
            sigmoid_function(vector_x.dot(theta)) - y_new)
        theta = theta - learningrate * gradients
        y_pred = sigmoid_function(vector_x.dot(theta))
        cost_value = -np.sum(
            np.dot(y_new.T,
                   np.log(y_pred) + np.dot(
                       (1 - y_new).T, np.log(1 - y_pred)))) / (len(y_pred))
        # Calculate the loss for each training instance
        cost_lst.append(cost_value)
    return theta


def column(matrix, i):
    """
    Returning all the values in a specific columns
        Parameters:
            X: the input matrix
            i: the column
        Return value: an array with desired column
    """
    return [row[i] for row in matrix]


def predict(x, theta):
    """
        Predict the outcome of the data set
    """
    hypo_line = theta[0]
    for i in range(1, len(theta)):
        hypo_line = hypo_line + theta[i] * column(x, i - 1)
    y_pred = sigmoid_function(hypo_line)

    for j in range(len(y_pred)):
        # print(j, y_pred[j])
        if y_pred[j] >= 0.5:
            y_pred[j] = 1
        else:
            y_pred[j] = 0

    return y_pred.reshape(len(y_pred), 1)

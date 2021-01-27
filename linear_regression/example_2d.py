""" In this module, we will extract the four fundamental notions for
supervised learning, through an example of linear regression on
two-dimensional data.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def model(data_x, parameters):
    """ This function returns the linear model

    """
    return data_x @ parameters


def cost_function(data_x, data_y, parameters):
    """ Returns the cost function for our model.

    """
    # Residual sum of squares
    rss = np.sum((data_x @ parameters - data_y) ** 2)

    # Number of rows times 2
    nb_rows_x2 = 2 * data_x.shape[0]

    return rss / nb_rows_x2


def gradient(data_x, data_y, parameters):
    """ Returns gradient of a function.

    """
    return data_x.T @ (data_x @ parameters - data_y) / data_x.shape[0]


def gradient_descent(data_x, data_y, parameters, learn_rate, nb_iterations):
    """ Finding a local minimum of a differentiable function by gradient
    descent algorithm.

    """

    for _i in range(nb_iterations):
        parameters -= learn_rate * gradient(data_x, data_y, parameters)

    return parameters


def main():
    """ Example of linear regression

    """
    ###################
    ##    Dataset    ##
    ###################
    # (X, y)  m = 100, n = 1
    data_x, data_y = make_regression(n_samples=100, n_features=1, noise=10)

    # show the data
    plt.scatter(data_x, data_y)

    # Writing the equations in the matrix form
    data_x = np.hstack((data_x, np.ones(data_x.shape)))
    data_y = data_y.reshape(data_y.shape[0], 1)

    #################
    ##    Model    ##
    #################
    # initial parameters
    init_params = np.random.randn(2, 1)

    # initial model
    init_model = model(data_x, init_params)

    # show initial model
    plt.plot(data_x, init_model, c='r')

    #########################
    ##    cost function    ##
    #########################
    # show cost function for initial params
    print(cost_function(data_x, data_y, init_params))

    ####################
    ##    training    ##
    ####################

    plt.show()


if __name__ == '__main__':
    main()

""" In this module, we will extract the four fundamental notions for
supervised learning, through an example of linear regression on
two-dimensional data.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def model(data_x, param):
    """ This function returns the linear model

    """
    return data_x @ param


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
    # parameter
    param = np.random.randn(2, 1)

    # initial model
    init_model = model(data_x, param)

    # show initial model
    plt.plot(data_x, init_model, c='r')

    #########################
    ##    cost function    ##
    #########################

    plt.show()


if __name__ == '__main__':
    main()

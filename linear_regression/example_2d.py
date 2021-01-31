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

    # Cost history
    cost_tracking = np.zeros(nb_iterations)

    for _i in range(nb_iterations):
        parameters -= learn_rate * gradient(data_x, data_y, parameters)
        # recording the cost for each iteration
        cost_tracking[_i] = cost_function(data_x, data_y, parameters)

    return parameters, cost_tracking


def coefficient_determination(data_y, prediction):
    """ Coefficient of determination.

    This function measures the quality of the prediction of linear regression.

    """
    sum_numerator = np.sum((data_y - prediction) ** 2)
    sum_denominator = np.sum((data_y - prediction.mean()) ** 2)
    return 1 - sum_numerator / sum_denominator


def main():
    """ Example of linear regression

    """
    ###################
    ##    Dataset    ##
    ###################
    # (X, y)  m = 100, n = 1
    _data_x, data_y = make_regression(n_samples=100, n_features=1, noise=10)

    # show the dataset
    plt.subplot(2, 2, 1)
    plt.title("dataset")
    plt.scatter(_data_x, data_y)

    # Transform the dataset into matrices.
    # That is used for writing the equations in the matrix form.
    data_x = np.hstack((_data_x, np.ones(_data_x.shape)))
    data_y = data_y.reshape(data_y.shape[0], 1)

    #################
    ##    Model    ##
    #################
    # initial parameters
    init_params = np.random.randn(2, 1)

    # initial model
    init_model = model(data_x, init_params)

    # plot initial model
    plt.subplot(2, 2, 2)
    plt.title("initial model")
    plt.scatter(_data_x, data_y)
    plt.plot(_data_x, init_model, c='g')

    #########################
    ##    cost function    ##
    #########################
    # show cost function for initial parameters
    print(cost_function(data_x, data_y, init_params))

    ####################
    ##    training    ##
    ####################
    # learning rate
    learn_rate = 0.005
    # number of iterations
    number_iterations = 1_000

    # final parameters for our model
    final_params, cost_tracking = gradient_descent(
        data_x, data_y, init_params, learn_rate, number_iterations)

    # final model
    final_model = model(data_x, final_params)

    # show cost function for final parameters
    print(cost_function(data_x, data_y, final_params))

    # plot final model
    plt.subplot(2, 2, 3)
    plt.title("final model")
    plt.scatter(_data_x, data_y)
    plt.plot(_data_x, final_model, c='r')

    ##########################
    ##    learning curve    ##
    ##########################
    # plot Cost history
    plt.subplot(2, 2, 4)
    plt.title("cost tracking")
    plt.plot(range(number_iterations), cost_tracking)

    ########################################
    ##    Coefficient of determination    ##
    ########################################
    print(coefficient_determination(data_y, final_model))

    plt.show()


if __name__ == '__main__':
    main()

#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Monty Hall problem.

Discover the Monty Hall problem, we visualization of results in a curve.
And search params of the curve, with machine learning,
we used linear regression.
"""


from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


# Two constants or two strategies used by the player (the player can
# choose one of them) in this game.

CHANGED = "Change"
KEEP = "Keep"


def play(strategy, n_times):
    """ Simulates a series of the Monty Hall game.

    """
    # The good door
    good_door = np.random.randint(0, 3, n_times)

    # The first choice of player
    first_choice = np.random.randint(0, 3, n_times)

    if strategy == CHANGED:
        # The presenter eliminates the wrong door
        return first_choice != good_door

    # second choice is equal to the first choice
    return first_choice == good_door


def main():
    """ main function

    """
    ############################################################################
    # Generate the data for our problem
    ############################################################################
    # List of numbers of turns in this game
    list_n_times = list(range(1, 10_000, 10))

    # In these lists, we will save the results of the simulation
    results_changed, results_keep = [], []

    for n_times in list_n_times:
        # The results of the strategy in which by changing the door
        results_changed.append(np.sum(play(CHANGED, n_times)))

        # The results of the strategy in which by keep the first choice
        results_keep.append(np.sum(play(KEEP, n_times)))

    # transform data from lists to arrays
    simple = np.array(list_n_times, dtype=np.float64)
    results_changed = np.array(results_changed, dtype=np.float64)
    results_keep = np.array(results_keep, dtype=np.float64)

    # Visualization of results
    # Plot the result with the scatter() function
    plt.scatter(simple, results_changed, label=CHANGED, c="#8CBBB1")
    plt.scatter(simple, results_keep, label=KEEP, c="#FFCC67")
    plt.legend()

    ############################################################################
    # search params of the curve, with curve_fit() function
    ############################################################################
    # This function returns the linear model
    model = lambda x, a, b: a * x + b

    # final parameters of our model
    params_changed, _ = optimize.curve_fit(model, simple, results_changed)
    params_keep, _ = optimize.curve_fit(model, simple, results_keep)

    # show and plot results
    print(params_changed)
    print(params_keep)
    plt.plot(simple, model(simple, *params_changed), c="r")
    plt.plot(simple, model(simple, *params_keep), c="r")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()

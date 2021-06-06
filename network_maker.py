#!/usr/bin/env python3
# network_maker.py
# Why shouldn't i try to write a NN from scratch, huh?
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

# For sigmoid
from numpy import array, exp


# Here goes nothing!
class Neuron:
    def __init__(self, weights: array, bias=0, activation_function=None):
        """
        Creates a neuron!!
        :param weights: A numpy array defining the weight of each input
        :param bias: The bias of this neuron, default is 0
        :param activation_function: The activation function!
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def calculate(self, inputs: array, weights=None) -> array:
        """
        Calculates the given inputs! Yes!
        :param inputs: A numpy array of the inputs!
        :return: A numpy array of the results!
        """
        # This is done in case weights are overriden
        if weights is None:
            weights = self.weights

        results = inputs * weights + self.bias

        if self.activation_function is not None:
            return self.activation_function(results)

        else:
            return results


def tanh(x):
    """
    tanh function.
    From https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
    """
    return (2 / (1 + exp(-2 * x))) - 1


def sigmoid(x):
    """
    Sigmoid function
    """
    # f(x) = 1/(1 + e^(-x))
    return 1/(1 + exp(-x))


if __name__ == "__main__":
    print("Try importing instead!")


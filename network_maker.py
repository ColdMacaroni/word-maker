#!/usr/bin/env python3
# network_maker.py
# Why shouldn't i try to write a NN from scratch, huh?
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

# Here goes nothing!

from numpy import array, zeros, exp, dot
from numpy.random import uniform


class HiddenLayer:
    def __init__(self, neuron_num, inputs,
                 weights=None, biases=None, activation_function=None):
        """
        Creates a hidden layer! Weights and biases randomized unless given!
        :param neuron_num: An integer representing the amount of neurons
        :param inputs: Int, the amount of inputs a neuron expects to receive

        :param weights: A 2d numpy array of weights for each neuron.
                        If this is not passed, they are assigned a
                        random weight between -1 and 1

        :param biases: A list of biases for each neuron
                       If this is not passed, they are assigned a
                       random bias between -1 and 1

        :param activation_function: The function for each neuron
        """
        if weights is None:
            # Generate random weights
            weights = uniform(-1, 1, neuron_num * inputs)

            # Reshape into 2D array
            weights.reshape((neuron_num, inputs))

        if biases is None:
            # Set em all to 0
            biases = zeros(neuron_num)

        # Create a set of neurons according to the starting weights
        self.neurons = set()
        for i in range(neuron_num):
            self.neurons.add(Neuron(weights[i], biases[i], activation_function))

    def process(self, inputs: array):
        """
        Takes an array of inputs and process them through the neurons!
        :param inputs: array
        :return: array
        """
        # Pass all the inputs through each neuron and generate an
        # array/matrix of the results
        results = array([neuron.process(inputs) for neuron in self.neurons])

        return results


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

    def process(self, inputs: array, weights=None) -> array:
        """
        Calculates the given inputs! Yes!
        :param inputs: A numpy array of the inputs!
        :param weights: A numpy array of weights in case they need a
                        temporary override
        :return: A numpy array of the results!
        """
        # This is done in case weights are overriden
        if weights is None:
            weights = self.weights

        results = dot(weights, inputs.T) + self.bias

        print("Neuron", results)

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


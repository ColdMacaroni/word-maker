#!/usr/bin/env python3
# network_nnfs.py
# Neural network objects following
# sentdex's neural networks from scratch videos

import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Creates an array of shape inputs, neurons.
        # Multiplies by 0.1 so all values are near 0
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



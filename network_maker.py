#!/usr/bin/env python3
# network_maker.py
# Why shouldn't i try to write a NN from scratch, huh?
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

# For sigmoid
from math import exp

# Here goes nothing!
class Neuron:
    def __init__():
        pass


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


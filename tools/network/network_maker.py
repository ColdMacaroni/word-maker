#!/usr/bin/env python3
# network_maker.py
# Why shouldn't i try to write a NN from scratch, huh?
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

# Here goes nothing!

from numpy import array, zeros, exp, dot, maximum, ndarray, full
from numpy.random import uniform
from json import dumps, loads


class Network:
    def __init__(self, n_inputs, n_layers, neurons_per_layer,
                 n_outputs, weights=None, biases=None,
                 activation_functions=None, output_act_func=None):
        """
        Create a neural network! Very nice.
        :param n_inputs: The amount of inputd
        :param n_layers: The amount of layers
        :param neurons_per_layer: A list of neurons per layer,
                                  can be an int for uniform layers
        :param n_outputs: The amount of outputs
        :param weights: A 3D array of the weights.
                        First dimension is the layers
                        Second dimension is the neuron
                        Third dimension is each input
        :param biases: A 2D array of the biases

        """
        self.output = None

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        self.n_outputs = n_outputs
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions
        self.output_act_func = output_act_func

        # I Wish this would Work.
        # These are the args that can be passed as a single value and
        # then be converted into a list of appropriate size
        # single_args = [
        #     self.weights,
        #     self.biases,
        #     self.neurons_per_layer,
        #     self.activation_functions
        # ]
        # for arg in single_args:
        #     if type(arg) != list:
        #         # Create a list of with the same values
        #         arg = [arg] * self.n_layers

        if type(self.neurons_per_layer) != list:
            # Create a list of with the same values
            self.neurons_per_layer = [self.neurons_per_layer] * self.n_layers

        if type(self.activation_functions) != list:
            # Create a list of with the same values
            self.activation_functions = [self.activation_functions] * self.n_layers
        
        if type(self.weights) != list:
            # Creates a list of with the same values

            # This list will store a 2d array of each layer's weights
            weights_ls = list()

            # Create the weights for the input layer
            # [[1] * 2 ] * 3 => [[1, 1], [1, 1], [1, 1]]
            if self.weights is None:
                # Add a 2D list of random weights
                weights_ls.append(
                    array([[uniform(-1, 1) for i in range(self.n_inputs)]
                           for n in range(self.neurons_per_layer[0])])
                )
            else:
                weights_ls.append(
                    array([[self.weights] * self.n_inputs]
                          * self.neurons_per_layer[0])
                )

            for layer in range(1, self.n_layers):
                if self.weights is None:
                    weights_ls.append(
                        # The amount of weights is the amount of last
                        # layer's neurons.
                        array([[uniform(-1, 1) for i in range(len(weights_ls[-1]))]
                               for n in range(self.neurons_per_layer[layer])])
                    )
                else:
                    weights_ls.append(
                        # The amount of weights is the amount of last
                        # layer's neurons.
                        array([[self.weights] * len(weights_ls[-1])]
                              * self.neurons_per_layer[layer])
                    )

            # Add the weights for the output
            if self.weights is None:
                # Add a 2D list of random weights
                weights_ls.append(
                    array([[uniform(-1, 1) for i in range(len(weights_ls[-1]))]
                           for n in range(self.n_outputs)])
                )
            else:
                weights_ls.append(
                    array([[self.weights] * len(weights_ls[-1])]
                          * self.n_outputs)
                )

            self.weights = weights_ls
            
        if type(self.biases) != list:
            # Create a list of with the same values
            biases_ls = list()

            # This way when initialized with None they
            # are started with 0
            repeat_bias = 0 if self.biases is None else self.biases

            # Create a 2d list for the biases
            # First dimension is each layer
            # Second dimension is each neuron
            # +1 for output layer
            for layer in range(self.n_layers):
                biases_ls.append([repeat_bias] * self.neurons_per_layer[layer])

            # Create the biases for the output layer
            biases_ls.append([repeat_bias] * self.n_outputs)

            self.biases = biases_ls

        assert len(self.neurons_per_layer) == self.n_layers, \
            "Discrepancy between neurons per layer and number of layers"

        # The +1 is there to count the output layer
        assert len(self.weights) == self.n_layers + 1, \
            "Discrepancy between weights and number of layers"

        assert len(self.biases) == self.n_layers + 1, \
            "Discrepancy between biases per layer and number of layers"

        self.layers = list()

        # Add Layer that receives input
        self.layers.append(
            HiddenLayer(
                self.neurons_per_layer[0], self.n_inputs,
                weights=self.weights[0], biases=self.biases[0],
                activation_function=self.activation_functions[0]
            )
        )

        for i in range(1, n_layers):
            self.layers.append(
                HiddenLayer(
                    self.neurons_per_layer[i],
                    # The amount of neurons of the prev layer == inputs
                    len(self.layers[-1].neurons),
                    weights=self.weights[i], biases=self.biases[i],
                    activation_function=self.activation_functions[i]
                )
            )

        # Add output layer
        self.layers.append(
            HiddenLayer(
                n_outputs,
                len(self.layers[-1].neurons),
                weights=self.weights[-1], biases=self.biases[-1],
                activation_function=output_act_func
            )
        )

    def forward(self, inputs: array):
        """
        Process the inputs!
        :param inputs: Array of inputs
        """
        # Transpose inputs to make dot products easier
        inputs = inputs.T
        for layer in self.layers:
            layer.forward(inputs)

            inputs = layer.output

        # Transpose back to a more readable format
        self.output = inputs.T

    def dump(self, filename=None, summary=False):
        """
        Dumps:
            n_inputs
            n_layers
            neurons_per_layer
            n_outputs
            activation_functions
            output_act_func
        Into the given file, this way they can be used anywhere.
        :param filename: The file to dump to
        :param summary: Print the dump then save
        """
        # Info
        if summary:
            print(f"Inputs\t\t\t\t{self.n_inputs}\n"
                  f"Layers\t\t\t\t{self.n_layers}\n"
                  f"Neurons per layer\t\t{self.neurons_per_layer}\n"
                  f"Outputs\t\t\t\t{self.n_outputs}\n"
                  f"Activation Functions\t\t{self.activation_functions}\n"
                  f"Output Activation function\t{self.output_act_func}\n")

        # Generate a dump_dict for this file (Or just a python dict)
        dump_dict = dict()

        # Create a list of the layers
        dump_dict["network"] = dict()

        # On each layer
        for layer in range(len(self.layers)):
            # Add key of the layer number with value of a dictionary
            # of that layer's neurons and the activation used
            dump_dict["network"][layer] = {"neurons": dict(),
                                           "activation": None}

            # On each neuron create a dict of key neuron's position
            # and value another dict with a list
            for neuron in range(len(self.layers[layer].neurons)):
                dump_dict["network"][layer]["neurons"][neuron] = {
                    "weights": list(self.weights[layer][neuron]),
                    "bias": self.biases[layer][neuron]
                }

            # -- Add the activation func
            # The last item is the output layer which has its function
            # stored in a different var
            if layer == len(self.layers) - 1:
                # Only get the name if the var is a function
                act_func_str = self.output_act_func.__name__\
                               if callable(self.output_act_func)\
                               else self.output_act_func

            else:
                # Only get the name if the var is a function
                act_func_str = self.activation_functions[layer].__name__\
                               if callable(self.activation_functions[layer])\
                               else self.activation_functions[layer]

            dump_dict["network"][layer]["activation"] = act_func_str

        dump_json = dumps(dump_dict)

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(dump_json)
        else:
            return dump_json


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
            weights = weights.reshape((neuron_num, inputs))

        if biases is None:
            # Set em all to 0
            biases = zeros(neuron_num)

        # Create a set of neurons according to the starting weights
        self.neurons = list()
        for i in range(neuron_num):
            self.neurons.append(Neuron(weights[i], biases[i], activation_function))

        self.output = None

    def forward(self, inputs: array):
        """
        Takes an array of inputs and process them through the neurons!
        :param inputs: array
        :return: array
        """
        # Pass all the inputs through each neuron and generate an
        # array/matrix of the results
        self.output = array([neuron.process(inputs) for neuron in self.neurons])


class Neuron:
    def __init__(self, weights: array, bias=0, activation_function=None):
        """
        Creates a neuron!!
        :param weights: A numpy array defining the weight of each input
        :param bias: The bias of this neuron, default is 0
        :param activation_function: The activation function!
        """
        self.weights = list()
        self.bias = 0 if bias is None else bias
        self.activation_function = activation_function

        # Generate a random value if it's none
        for weight in weights:
            if weight is None:
                self.weights.append(uniform(-1, 1))

            else:
                self.weights.append(weight)

        # Convert to array for easier dot products
        self.weights = array(self.weights)

    def process(self, inputs: array) -> array:
        """
        Calculates the given inputs! Yes!
        :param inputs: A numpy array of the inputs!
        :return: A numpy array of the results!
        """
        results = dot(self.weights, inputs) + self.bias

        if self.activation_function is not None:
            return self.activation_function(results)

        else:
            return results


def ReLU(x):
    """
    Rectified linear
    :param x:
    :return:
    """
    return maximum(0, x)


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
    return 1 / (1 + exp(-x))


if __name__ == "__main__":
    # This is only used here, no need to add weight when imported
    from random import randint

    print("test_network = Network(2, 5, 10, 1, activation_functions=ReLU)")
    test_network = Network(2, 5, 10, 1, activation_functions=ReLU)

    #test_data = uniform(-1, 1, (2, 2))
    test_data = array([[0.6, 0.4], [-0.2, 1]])

    print(f"test_network.forward(\n{test_data}\n)")
    test_network.forward(test_data)

    print()

    print("Output:")
    print(str(test_network.output))

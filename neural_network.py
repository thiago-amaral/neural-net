import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture

        '''
        Creating a weight matrix for each layer.
        Includes biases.
        (Obviously excludes output layer).
        The weights are initialized randomly.
        '''
        self.weights = [np.random.randn(architecture[i + 1], architecture[i] + 1)
                        for i in range(len(architecture) - 1)]

        # Creating a vector of neurons for each layer (initialized with zeros).
        self.neurons = [np.zeros(architecture[i])
                        for i in range(len(architecture))]

    def mean_squared_cost(self, x, y):
        # Mean squared used only to estimate cost.
        # Gradients are being computed with log loss.
        cost = 0

        for i in range(len(x)):
            cost += np.sum(y[i] - self.feed_forward(x[i])) ** 2

        return cost / len(x)

    def __activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        self.neurons[0] = x

        '''
        Each new activation vector is equal to the previous
        weights multiplied by the previous activations.
        Inserts 1 to calculate with biases.
        '''
        for i in range(1, len(self.neurons)):
            self.neurons[i] = self.__activation(
                self.weights[i - 1] @ np.insert(self.neurons[i - 1], 0, 1))

        return self.neurons[len(self.neurons) - 1]

    def backprop(self, x, y, alpha, reg):
        '''
        Computes delta for each layer, and then update each weight.
        Uses one training example. (Stochastic Gradient Descent)
        '''
        archi = self.architecture

        y_hat = self.feed_forward(x)

        delta = {}

        # Error (delta) in the last layer is output minus label.
        delta[len(archi) - 1] = (y_hat - y).reshape(archi[len(archi) - 1], 1)

        # Computing deltas up to last hidden layer.
        for i in reversed(range(1, len(archi) - 1)):
            activ = np.insert(self.neurons[i], 0, 1).reshape(archi[i] + 1, 1)

            delta[i] = np.delete(
                (self.weights[i].T @ delta[i + 1]) * activ * (1 - activ), [0], axis=0)

        # Updating weights: Gradient Descent.
        for i in range(len(self.weights)):
            activ = np.insert(self.neurons[i], 0, 1).reshape(archi[i] + 1, 1)

            gradient = delta[i + 1] @ activ.T

            # First column have the biases, it shouldn't be regularized.
            self.weights[i][:, 1:] -= alpha * \
                gradient[:, 1:] + (reg * self.weights[i][:, 1:])

            self.weights[i][:, 0] -= alpha * gradient[:, 0]

    def train(self, x, y, alpha, epochs, reg, debug=False):
        cost = []

        for _ in range(epochs):
            for i in range(len(x)):
                self.backprop(x[i], y[i], alpha, reg)

            cost.append(self.mean_squared_cost(x, y))

        plt.plot(cost)
        plt.show()

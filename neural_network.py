"""
neural_network.py - Lightweight Feedforward Neural Network with Genetic Mutation support.
No external ML frameworks required; uses standard math and random.
"""
import math
import random
import json


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of ints defining each layer, e.g. [5, 10, 4, 2]
        """
        self.layer_sizes = list(layer_sizes)
        self.weights = []
        self.biases = []
        self.last_activations = []

        for l in range(len(layer_sizes) - 1):
            rows = layer_sizes[l]
            cols = layer_sizes[l + 1]
            # Xavier initialization
            limit = math.sqrt(2.0 / (rows + cols))
            w_layer = [
                [random.uniform(-limit, limit) for _ in range(cols)]
                for _ in range(rows)
            ]
            b_layer = [random.uniform(-0.1, 0.1) for _ in range(cols)]
            self.weights.append(w_layer)
            self.biases.append(b_layer)

    def forward(self, inputs):
        """Forward propagation. Returns list of output floats."""
        activations = [list(inputs)]
        current = list(inputs)

        for l in range(len(self.weights)):
            w = self.weights[l]
            b = self.biases[l]
            rows = len(w)
            cols = len(w[0])
            next_layer = []

            for j in range(cols):
                total = b[j]
                for i in range(rows):
                    total += current[i] * w[i][j]
                val = math.tanh(total)
                next_layer.append(val)

            current = next_layer
            activations.append(current)

        self.last_activations = activations
        return current

    def clone(self):
        """Creates an exact independent copy."""
        child = NeuralNetwork(self.layer_sizes)
        child.weights = [[list(row) for row in layer] for layer in self.weights]
        child.biases = [list(layer) for layer in self.biases]
        return child

    def mutate(self, rate=0.08, magnitude=0.3):
        """Mutates weights and biases with Gaussian noise."""
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][0])):
                    if random.random() < rate:
                        self.weights[l][i][j] += random.gauss(0, magnitude)
                        self.weights[l][i][j] = max(-5.0, min(5.0, self.weights[l][i][j]))

            for j in range(len(self.biases[l])):
                if random.random() < rate:
                    self.biases[l][j] += random.gauss(0, magnitude)
                    self.biases[l][j] = max(-3.0, min(3.0, self.biases[l][j]))

    def crossover(self, partner):
        """Uniform crossover with partner to produce a child."""
        child = self.clone()
        for l in range(len(child.weights)):
            for i in range(len(child.weights[l])):
                for j in range(len(child.weights[l][0])):
                    if random.random() < 0.5:
                        child.weights[l][i][j] = partner.weights[l][i][j]
            for j in range(len(child.biases[l])):
                if random.random() < 0.5:
                    child.biases[l][j] = partner.biases[l][j]
        return child

    def save(self, filepath):
        """Saves network to JSON."""
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Loads a network from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        nn = cls(data["layer_sizes"])
        nn.weights = data["weights"]
        nn.biases = data["biases"]
        return nn

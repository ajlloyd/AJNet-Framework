import numpy as np
from sklearn.datasets import make_classification
np.random.seed(42)

x_train, y_train = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=0, n_classes=2)

#-------------------------------------------------------------------------------

class Layers:
    def Dense(nodes=None):
        return nodes

class Frame():
    def __init__(self):
        self.model_list = []

    def add(self, layer):
        self.model_list.append(layer)

    def initialise_w_and_b(self, x, y):
        layers = len(self.model_list)
        nodes_per_layer = self.model_list
        weights = []
        biases = []
        for layer, nodes in zip(range(layers), nodes_per_layer):
            if layer == 0:
                weights.append(np.random.normal(loc=0, scale=1, size=(x.shape[1], nodes)))
            else:
                prev_nodes = weights[layer-1].shape[1]
                weights.append(np.random.normal(loc=0, scale=1, size=(prev_nodes, nodes)))
            biases.append(np.zeros((1,nodes)).ravel())
        return weights, biases




model = Frame()
model.add(Layers.Dense(nodes=5))
model.add(Layers.Dense(nodes=4))
model.add(Layers.Dense(nodes=2))
model.add(Layers.Dense(nodes=1))

print(model.initialise_w_and_b(x_train, y_train))

#model.run(iterations=1000, learning_rate=0.01)

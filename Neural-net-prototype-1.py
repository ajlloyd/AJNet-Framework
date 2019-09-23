
import numpy as np
from sklearn.datasets import make_classification
np.random.seed(42)

x_train, y_train = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2)

#-------------------------------------------------------------------------------

hidden_layers = 3
nodes = [4, 2, 1]

def initialize_weights(x, layers=None, nodes_per_layer=None):
    weights = []
    for layer, nodes in zip(range(layers), nodes_per_layer):
        if layer == 0:
            weights.append(np.random.normal(loc=0, scale=1, size=(x.shape[1], nodes)))
        else:
            prev_nodes = weights[layer-1].shape[1]
            weights.append(np.random.normal(loc=0, scale=1, size=(prev_nodes, nodes)))
    return weights
layer_weights = initialize_weights(x_train, layers=hidden_layers, nodes_per_layer=nodes)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sig_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def backprop_output(output_a, previous_a, y):
    output_delta = (output_a - y.reshape(-1,1))
    dc_dw = np.dot(output_delta.T, previous_a).reshape(-1,1)
    return dc_dw, output_delta

def backprop_hidden(bp_a_delta, bp_w ,current_z, previous_a, y):
    dc_da = np.dot(bp_a_delta, bp_w.T)              # Eq.1
    da_dz = sig_der(current_z)                      # Eq.2
    dz_dw = previous_a                              # Eq.3
    a_delta = (dc_da * da_dz)                       # Eq.1 * Eq.2
    dc_dw = np.dot(dz_dw.T, a_delta)                # Eq.2 dot a_delta
    return dc_dw, a_delta


def feed_forward(weights, x, y):
    a = x
    z_vals = [] # z(w.a) for each layer
    a_vals = [] # activations for each layer
    for w in weights:
        z = np.dot(a,w)
        a = sigmoid(z)
        z_vals.append(z)
        a_vals.append(a)
    return a_vals, z_vals


def run(starting_w, x, y, iterations=1):
    w = starting_w

    for i in range(iterations):
        a, z = feed_forward(w, x, y)
        input = x
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]

        z1 = z[0]
        z2 = z[1]
        z3 = z[2]

        a1 = a[0]
        a2 = a[1]
        a3 = a[2]

        output_bp = backprop_output(a3, a2, y)
        w3_update = output_bp[0]
        print(w3_update.mean()) #####
        a3_delta = output_bp[1]


        hidden2_bp = backprop_hidden(a3_delta, w3 ,z2, a1, y)
        w2_update = hidden2_bp[0]
        a2_delta = hidden2_bp[1]


        hidden1_bp = backprop_hidden(a2_delta, w2, z1, input, y)
        w1_update = hidden1_bp[0]
        a1_delta = hidden1_bp[1]


        w3 -= 0.1 * w3_update
        w2 -= 0.1 * w2_update
        w1 -= 0.1 * w1_update

        w = [w1, w2, w3]
    print(a3.round(decimals=2))

run(layer_weights, x_train, y_train, iterations=1000)

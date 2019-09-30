import numpy as np
from sklearn.datasets import make_classification
np.random.seed(42)

x_train, y_train = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=0, n_classes=2)

#-------------------------------------------------------------------------------

class Layers():
    def Dense(nodes=None):
        return nodes



class Activations:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def sig_der(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))



class Frame(Activations):
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

    def backprop_hidden(self, bp_a_delta, bp_w ,current_z, previous_a, y):                # Hidden layer Backprop derivatives:
        dc_da = np.dot(bp_a_delta, bp_w.T)                                          # Eq.1
        da_dz = super().sig_der(current_z)                                          # Eq.2
        dz_dw = previous_a                                                          # Eq.3
        a_delta = (dc_da * da_dz)                                                   # Eq.1 * Eq.2
        dc_dw = np.dot(dz_dw.T, a_delta)                                            # Eq.2 dot a_delta
        return dc_dw, a_delta

    def backprop_output(self, output_a, previous_a, y):                                   # Output layer derivates (different)
        output_delta = (output_a - y.reshape(-1,1))                                 # o_delta == error
        dc_dw = np.dot(output_delta.T, previous_a).reshape(-1,1)
        print(dc_dw.mean())                                                         ##### gradient (REMOVE) #####
        return dc_dw, output_delta

    def feed_forward(self, weights, biases, x, y):
        a = x
        z_vals = []                                                                 # Z for each layer [Z = (w.a) + b] (a = input for layer 0)
        a_vals = []                                                                 # activations for each layer [A = activation(Z)]
        for w, b in zip(weights, biases):
            z = np.dot(a,w) + b
            a = super().sigmoid(z)
            z_vals.append(z)
            a_vals.append(a)
        return a_vals, z_vals

    def run(self, x, y, iterations=1, learning_rate = 0.1):
        w = self.initialise_w_and_b(x, y)[0]
        b = self.initialise_w_and_b(x, y)[1]
        for i in range(iterations):
            a, z = self.feed_forward(w, b, x, y)                                         # feed_forward pass function
            delta = None                                                            # delta val which is updated for each layer
            w_updates = []                                                          # w and b updates
            b_updates = []
            for layer_n in reversed(range(len(w))):
                input = x
                current_w = [w_n for w_n in reversed(w)]                            # these lines reverse the w, a, and z values from the feed_forward pass
                current_a = [a_n for a_n in reversed(a)]                            # layers need to be reversed as the lower layer
                current_z = [z_n for z_n in reversed(z)]                            # calculations rely on higher layer ones (i.e. delta)
                # For Output Hidden Layer:
                if layer_n == (len(w)-1):
                    output_bp = self.backprop_output(current_a[0], current_a[1], y)
                    w_updates.append(output_bp[0])
                    delta = output_bp[1]
                    b_updates.append(np.array(delta))
                # For Middle Hidden Layers:
                elif layer_n != (len(w)-1) and layer_n != 0:
                    hidden_layer_bp = self.backprop_hidden(delta, current_w[-(layer_n + 2)] ,current_z[-(layer_n + 1)], current_a[-layer_n], y)
                    w_updates.append(hidden_layer_bp[0])
                    delta = hidden_layer_bp[1]
                    b_updates.append(np.array(delta))
                # For the Initial Hidden Layer:
                elif layer_n == 0:
                    initial_bp = self.backprop_hidden(delta, current_w[-(layer_n + 2)], current_z[-(layer_n + 1)], input, y)
                    w_updates.append(initial_bp[0])
                    delta = initial_bp[1]
                    b_updates.append(np.array(delta))
            summed_b_updates = [sum(bs) for bs in b_updates]                        # sums of all the delta values for each instance (this sum is the amount b is updated by)
            w_update_arr = np.array(w_updates)[::-1]                                # these lines re-reverse the layers (so that these arrays are now the orignial way round)
            b_update_arr = np.array(summed_b_updates)[::-1]
            w = np.array(w)                                                         # these lines convert w/b (list) from previous iteration to w/b (np.array) so that they may be updated as per below
            b = np.array(b)
            w -= learning_rate * w_update_arr                                       # these two lines update the weights and biases for each iteration
            b -= learning_rate * b_update_arr

        #print(np.c_[a[-1].round(decimals=2), y])
        final_a = a[-1]
        return final_a

#-------------------------------------------------------------------------------


model = Frame()
model.add(Layers.Dense(nodes=5))
model.add(Layers.Dense(nodes=4))
model.add(Layers.Dense(nodes=2))
model.add(Layers.Dense(nodes=1))

#model.initialise_w_and_b(x_train, y_train)

model.run(x_train, y_train, iterations=1000, learning_rate=0.01)

import os
import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)
print(os.getcwd())

x_train, y_train = make_classification(n_samples=20, n_features=3, n_informative=3, n_redundant=0, n_classes=2)

#-------------------------------------------------------------------------------

class Layers:
    def Dense(nodes=None, activation=None):
        return nodes, activation





class Activation:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        z = np.array([max(0,val) for row in z for val in row]).reshape(-1, z.shape[1])
        return z

    def leaky_relu(self, z):
        z = np.array([max(val*0.01,val) for row in z for val in row]).reshape(-1, z.shape[1])
        return z

    def select(self,string, z):
        if string == "sigmoid":
            return self.sigmoid(z)
        elif string == "relu":
            return self.relu(z)
        elif string == "leaky_relu":
            return self.leaky_relu(z)
        else:
            return None




class Frame(Activation):
    def __init__(self):
        self.model_list = []

    def add(self, layer):
        self.model_list.append(layer)

    def return_model(self):
        # return the model_list produced with Frame.add
        return self.model_list

    def initialise(self, x, loss):
        model_list = self.return_model()

        for layer in model_list:
            w = np.random.rand(x.shape[1], nodes)


        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        print(f"Model used: {model_list}")
        print(f"Loss Fucntion: {loss}")
        return x_bias


    def training(self, x, y, epochs=10, iterations=1000, loss="MSE"):
        model = self.return_model()                 #model to use (loss is MSE for now, optimiser is GD)
        x = self.initialise(x, loss)                      #x to use

        """for e in range(epochs):
            for i in range(iterations):
                print(f"Epoch:{e+1}, Iteration:{i}")
                for layer in model:"""



        """#for e in epochs:
        for i in range(iterations):
            for layer in model:
                nodes = layer[0]
                activation_string = layer[1]
                w = np.random.rand(x.shape[1], nodes)
                z = x.dot(w)
                pred_y = super().select(activation_string, z)

                error = y - pred_y                          # error (actual - predicted)
                adjustment = error * sig_dev(pred_y)
                w += np.dot(input_layer.T, adjustment)      # update weights"""






model = Frame()
model.add(Layers.Dense(nodes=2, activation="sigmoid"))
model.add(Layers.Dense(nodes=3, activation="sigmoid"))

model.training(x_train, y_train, epochs=1, iterations=100)

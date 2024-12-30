
import numpy as np
from matplotlib  import pyplot as plt



def sigmoid(Z):
        return 1.0/(1.0+np.exp(-Z))

def sigmoid_prime(Z):
        return sigmoid(Z)*(np.ones_like(Z)-sigmoid(Z))

def vector(y):
    Y = []
    for value in y:
        li = [0. for k in range(10)]
        li[value] = 1.
        Y.append(li)
    Y = np.array(Y).T
    return Y



class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    
    def forward_prop(self, input):
        a = input
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backpropagation(self, X, Y):
        L = self.num_layers - 1
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward
        activation = X
        activations = [X] #liste pour stocker tous les a, couche par couche
        zs = [] #liste pour stocker tous les z, couche par couche
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) +b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = (activations[-1]- vector(Y))* sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, L):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_w, nabla_b)

    def cost(self, batch):
        sum = 0
        for X, y in batch:
            Y = vector(y)
            sum+= 1/len(batch)* np.linalg.norm(self.forward_prop(X)-Y)**2
        return sum


    def moy(self, batch):
        sum = 0
        for X, y in batch: 
            if y[0] == np.argmax(self.forward_prop(X)):
                sum += 1
        return sum/len(batch)

    def update(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for X, Y in batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(X, Y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(batch))*nw for w, nw in zip(self.biases, nabla_w)]
        self.biases = [b - (eta/len(batch))*bw for b, bw in zip(self.weights, nabla_b)]
    
    def evaluate(self, test_data):
        return self.moy(test_data)
    
    def gradient_descent(self, training_data, epochs, eta, test_data = None):
        for j in range(epochs):
           self.update(training_data, eta)
           print("Epoch {} complete".format(j+1))
           if test_data:
            print("Moyenne sur le set de test : {}".format(self.evaluate(test_data)))

    def SGD(self, training_data, epochs, eta, test_data = None):
        for j in range(epochs): 
            for X, Y in training_data:
                self.update([(X, Y)], eta)
            print("Epoch {} complete".format(j+1))
            if test_data:
                print("Moyenne sur le set de test : {}".format(self.evaluate(test_data)))


    def train_mini_batch(self, training_data, epochs, size_mini_batch, eta, test_data = None):
        n = len(training_data)
        if test_data:
            print("Moyenne sur le set de test : {}".format(self.evaluate(test_data)))
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k + size_mini_batch] for k in range(0, n, size_mini_batch)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
            print("Epoch {} complete".format(j+1))
            if test_data:
                print("Moyenne sur le set de test : {}".format(self.evaluate(test_data)))
                        

        


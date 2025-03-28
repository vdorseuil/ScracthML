import numpy as np
import pandas as pd
from matplotlib  import pyplot as plt

from model import Network

data_train = pd.read_csv("mnist_train.csv")
data_test = pd.read_csv("mnist_test.csv")

data_train = np.array(data_train)
data_test = np.array(data_test)


m, n = data_train.shape
data_train = data_train.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255

data_test = data_test.T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test/255


def vector(y):
    Y = []
    for value in y:
        li = [0. for k in range(10)]
        li[value] = 1.
        Y.append(li)
    Y = np.array(Y).T
    return Y


def batch(X, Y):
    li = []
    for k in range(len(X[0])):
        li.append((X[:, k:k+1], Y[k:k+1]))
    return li

train_data = batch(X_train, Y_train)
test_data = batch(X_test, Y_test)

liste_layer = [784, 16, 16, 10]



if __name__ == "__main__":
    net = Network(liste_layer)
    net.train_mini_batch(train_data[0:6000], 100, 100, 0.1, test_data)

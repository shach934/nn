# this is a module for neural network
# All the arrays in the module is numpy array. avoid using native python3 list.

import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle

def ReLU(A):
    # Rectified Linear Unit, R(x) = x, if x > 0, R(x) = 0, if x <= 0
    return np.max(A, 0)

def sigmoid(Z):
    # sigmoid func sig(1 / (1 + exp(-z)))
    return 1 / (1 + np.exp(- Z))

def softmax(Z):
    # calculate the softmax of Z.
    a = np.exp(Z)
    return a / np.sum(a, axis = 0)

def random_initialize(dimension):
    # randomly initilize the weight matrix
    # alpha: the hyperparameter, put a small number to make sure in the middle ragion 
    alpha = 1
    np.random.seed(1)
    return np.random.randn(dimension[0], dimension[1]) * alpha

def initialize_W_b(inputSize, netStruc):
    # randomly initialize the W, b is intialized to 0
    N_layers = len(netStruc)
    firstLayer = (netStruc[0], inputSize)
    W, b = [], []
    W.append(random_initialize(firstLayer))
    for i in range(1, N_layers):
        dimension = (netStruc[i], netStruc[i - 1])
        W.append(random_initialize(dimension))
        b.append(np.random.randn(netStruc[i], 1))
    return W, b

def forward(input, label, W, b):

 # forward propagation
 # input:   the train samples, of dimension n_x by m,
 # n_x :    length of input, 
 # m :      number of training samples.
 # label:   the labels for the train samples, dimension m by 1

 # output is the cost func J, cached z values, the forward result p of dimension m by 1

 # note: the last layer is softmax to classify multiply catagories.

    m = input.shape[1]
    L = len(W)
    cache = []
    A = input
    for layer in range(L - 1):
        A = np.dot(W[layer], A) + b[layer]
        cache.append((A, W))
        A = ReLU(A)

    # the last layer is sigmoid to classify type
    Z_L = np.dot(W[L - 1], A) + b[L - 1]
    cache.append(Z_L)
    y = softmax(Z_L)
    J = -sum(sum(y - label)) / m
    # sigmoid cost function
    # J = -sum(np.multiply(y, np.log(label)) + np.multiply(1 - y, np.log(1 - label))) / m
    return J, y, cache
 
def backward(J, cache, label, W, b):
    # back propogation
    # calculate the partial differential dw and db

    # default the last layer is sigmoid 
    # in other layers the activation func is ReLU

    # cache : the (Z, W) during forward is cached. length L.
    
    L = len(W)
    dW, db = W, b
    dZ = cache[-1] - label # the last layer is softmax, so the dz_L = a_L - y
    for i in range(L - 1, 0, -1):
        dW[i] = np.dot(dZ, cache[i][0].T)
        db[i] = dZ
        dZ = np.dot(cache[i][1].T, dZ) * (cache[i-1][0] > 0)
    dW[0] = np.dot(dZ, cache[i][0].T)
    db[0] = dZ
    return dW, db

def nn(input, netStruc, label, alpha, iters, showCost):
# input:    the training samples, of dimension n_x  by m
# label:    the lable for corresponding training samples, of dimension 1 by m
# netStruc: the structure of neural network specified by the neurons in each layers, 
# like [2, 3, 4, 5, 1], five layers in total and 2, 3, 4, 5, 1 neurons in each layer

    n_x = input.shape[0]
    W, b = initialize_W_b(n_x, netStruc)
    for i in range(iters):
        J, y, cache = forward(input, label, W, b)
        dW, db = backward(J, cache, label, W, b)
        W = W - alpha * dW
        b = b - alpha * db
        if showCost:
            plt.plot(i, J)
            plt.show()
    return W, b

def predict(sample, label, W, b):
    y_hat = forward(sample, label, W, b)
    m = label.shape[1]
    y = np.max(y_hat) == y_hat
    return sum(sum(y)) / m * 100.0

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

train_sample, train_label = train_set[0], train_set[1]

iters = 100
showCost = True
netStruc = [20, 50, 30, 10]
alpha = 0.1
for i in range(iters):
    W, b = nn(train_sample, netStruc, train_label, alpha, iters, showCost)
    accu = predict(test_set[0], test_set[1], W, b)
    print(accu)

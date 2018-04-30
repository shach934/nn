# this is a module for neural network
# All the arrays in the module is numpy array. avoid using native python3 list.

import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle

def ReLU(A):
    # Rectified Linear Unit, R(x) = x, if x > 0, R(x) = 0, if x <= 0
    return np.maximum(A, 0)

def sigmoid(Z):
    # sigmoid func sig(1 / (1 + exp(-z)))
    return 1 / (1 + np.exp(- Z))

def softmax(Z):
    # calculate the softmax of Z.
    a = np.exp(Z)
    return a / np.sum(a, axis = 0)

def initialize_W_b(inputSize, netStruc):
    # randomly initialize the W, b is intialized to 0
    N_layers = len(netStruc)
    W, b = [], []
    np.random.seed(0)
    alpha = 0.1
    W.append(np.random.randn(netStruc[0], inputSize) * alpha)
    b.append(np.random.randn(netStruc[0], 1))
    for i in range(1, N_layers):
        W.append(np.random.randn(netStruc[i], netStruc[i - 1]) * alpha)
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
        # deep copy, or cached A will change and only the last layer A be saved. 
        cache.append((A.copy(), W[layer].copy()))   
        A = ReLU(A)                       

    # the last layer is sigmoid to classify type
    Z_L = np.dot(W[L - 1], A) + b[L - 1]
    cache.append((Z_L.copy(), W[L-1].copy()))
    y = softmax(Z_L)
    J = -(y - label).sum() / m
    # sigmoid cost function
    # J = -sum(np.multiply(y, np.log(label)) + np.multiply(1 - y, np.log(1 - label))) / m
    return J, y, cache
 
def backward(J, cache, label, W, b, input):
    # back propogation
    # calculate the partial differential dw and db

    # default the last layer is sigmoid 
    # in other layers the activation func is ReLU

    # cache : the (Z, W) during forward is cached. length L.
    
    L = len(W)
    dW, db = [], []
    dZ = cache[-1][0] - label    # the last layer is softmax, so the dz_L = a_L - y
    for i in range(L - 1, 0, -1):
        dW.append(dZ.dot(cache[i - 1][0].T))
        db.append(dZ)
        dZ = np.dot(cache[i][1].T, dZ) * (cache[i-1][0] > 0)
    dW.append(np.dot(dZ, input.T))
    db.append(dZ)
    return dW, db

def nn(input, netStruc, label, alpha, iters, showCost):
# input:    the training samples, of dimension n_x  by m
# label:    the lable for corresponding training samples, of dimension 1 by m
# netStruc: the structure of neural network specified by the neurons in each layers, 
# like [2, 3, 4, 5, 1], five layers in total and 2, 3, 4, 5, 1 neurons in each layer

    n_x = input.shape[0]
    W, b = initialize_W_b(n_x, netStruc)
    I, J = [], []
    for i in range(iters):
        j, y, cache = forward(input, label, W, b)
        dW, db = backward(J, cache, label, W, b, input)
        W = [w  - alpha*dw  for w,  dw  in zip(W, dW)]
        b = [bb - alpha*dbb for bb, dbb in zip(b, db)]
        I.append(i)
        J.append(j)
        if showCost:
            plt.plot(I, J)
            plt.show()
    return W, b

def predict(sample, label, W, b):
    y_hat = forward(sample, label, W, b)
    m = label.shape[1]
    y = np.max(y_hat) == y_hat
    return y.sum() / m * 100.0

def expandLabel(m, label2ex):
    label = np.zeros((10, m))
    j = np.arange(m)
    label[label2ex, j] = 1
    return label

def loadMNIST():
    # the formal program, the size don't fit, the input is rank arrary, try to reshape it.
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    train_sample, train_label = train_set[0].T, train_set[1].reshape(1, -1)
    valid_sample, valid_label = valid_set[0].T, valid_set[1].reshape(1, -1)
    test_sample,  test_label  = test_set[0].T,  test_set[1].reshape(1, -1)

    m_train = train_sample.shape[1]
    m_valid = valid_sample.shape[1]
    m_test  = test_sample.shape[1]

    train_label = expandLabel(m_train, train_label)
    valid_label = expandLabel(m_valid, valid_label)
    test_label  = expandLabel(m_test,  test_label)
    """
    print("Train sample shape: ", train_sample.shape)
    print("Train label shape: ", train_label.shape)
    print("valid sample shape: ", valid_sample.shape)
    print("valid label shape: ", valid_label.shape)
    print("test sample shape: ", test_sample.shape)
    print("test label shape: ", test_label.shape)
    """
    return (train_sample, train_label), (valid_sample, valid_label), (test_sample, test_label)

train, valid, test = loadMNIST()

iters = 100
showCost = True
netStruc = [20, 10]
alpha = 0.1

train_sample = train[0]
train_label  = train[1]

W, b = nn(train_sample, netStruc, train_label, alpha, iters, showCost)
accu = predict(test[0], test[1], W, b)
print(accu)
# neural network

# First version: all functions, later may pack it to a class.

import numpy as np
import matplotlib.pyplot as plt

def ReLU(Z):
    return np.max(0, Z)
    
def sigmoid(Z):
    return 1 / (1 - np.exp(-Z))
  
def initilizeW(dimension):
    return np.random.randn(dimension)

def forward(input, net_struct):
    L = len(net_struct)
    W = []
    a0 = len(input)
    for i in range(L - 1):
        dimension = (net_struct(i))
        W.append(initilizeW(net_struct[i]))
        




net_struct = [4,]
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

def forward(inputSize, net_struct):
    # the first in the net_struct is the size of input. 
    # then the number of neurals in the following layers.
    L = len(net_struct)
    W = []
    for i in range(1, L):
        dimension = (net_struct(i - 1), net_struct(i))
        W.append(dimension)
    b = np.zeros(())
        




net_struct = [4,]
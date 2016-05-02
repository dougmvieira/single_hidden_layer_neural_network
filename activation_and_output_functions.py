"""
Activation and output functions for the neural network and their derivatives.

"""

import numpy as np

def argmax_classifier(p):
    return np.argmax(p, axis=1)

def softmax(x, k):
    """ Evaluates the softmax function used as the output function of a neural
    network for classification, namely, softmax(x, k) = exp(x[k])/sum(exp(x))
    or softmax(x, k) = exp(x[k] + max(x))/sum(exp(x + max(x))). The latter
    version is preferred for numerical stability. If `x` is a NxK matrix and
    `k` is an integer, the sigmoid function is evaluated row-wise and return an
    N-sized vector. If, however, `k` is an M-dimensional vector, the function
    returns an NxM matrix.
    """
    if x.ndim == 1:
        max_x = np.max(x)
        return np.exp(x[k] + max_x)/np.sum(np.exp(x + max_x))
    else:    
        max_x = np.expand_dims(np.max(x, axis=1), axis=1)
        return (np.exp(x[:, k] + max_x)
                / np.expand_dims(np.sum(np.exp(x + max_x), axis=1), axis=1))

def sigmoid(x):
    """ Evaluates the sigmoid function used as the activation function in
    neural networks, namely sigmoid(x) = 1/(1 + exp(-x)) = exp(x)/(exp(x) + 1).
    Both versions are used for numerical stability. If `x` is a vector or a
    matrix, the sigmoid function is evaluated element-wise.
    """
    x_plus = np.maximum(0., x)
    x_minus = x_plus - x
    return 1./(1. + np.exp(x_minus)) + np.exp(x_plus)/(np.exp(x_plus) + 1) - .5

def derivative_of_sigmoid(x):
    """ Evaluates the derivative of the sigmoid function, namely,
    derivative_of_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)). If `x` is a
    vector or a matrix, the function is evaluated element-wise.
    """
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def ReLU(x):
    """ Evaluates the ReLU function used as the activation function in neural
    networks, namely ReLU(x) = max(0, x). If `x` is a vector or a matrix, the
    ReLU function is evaluated element-wise.
    """
    return np.maximum(0., x)

def derivative_of_ReLU(x):
    """ Evaluates the (right) derivative of the ReLU function, namely,
    derivative_of_ReLU(x) = 1 if x >= 0 or 0 if x < 0. If `x` is a vector or a
    matrix, the function is evaluated element-wise.
    """
    return (np.array(x) >= 0.).astype(float)
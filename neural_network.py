"""
Single hidden layer neural network implemented using back propagation.

"""

import numpy as np
from gradient_descent import gradient_descent, mini_batch_gradient_descent
from activation_and_output_functions import (argmax_classifier, softmax,
                                             sigmoid, derivative_of_sigmoid,
                                             ReLU, derivative_of_ReLU)

def back_propagation_delta(y, y_tilde):
    """ Back propagation matrix delta which, in this case, is the Jacobian
    of the cross-entropy loss function with respect to Z, the output of the
    hidden layer, where the output function is the softmax operator.
    """
    return y_tilde - y

def back_propagation_s(delta, w, beta, sigma_prime):
    """ Back propagation matrix s which, in this case, is the Jacobian
    of the output of the hidden layer with respect to the input X.
    """
    return sigma_prime(w) * np.dot(delta, np.transpose(beta))

def neural_network(x, g, sigma, alpha_0, alpha, beta_0, beta):
    """ Evaluates the neural network with one hidden layer.

    Args:
        x (1d or 2d Numpy array): Input vector x.
        g (function: array-like to array-like): Output function (e.g. softmax
            function).
        sigma (function: array-like to array-like): Activation function (e.g.
            sigmoid function).
        alpha_0 (NumPy array): Bias parameter vector from the input layer to
            the hidden layer.
        alpha (NumPy 2d array): Weights parameter matrix from the input layer
            to the hidden layer.
        beta_0 (NumPy array): Bias parameter vector from the hidden layer to
            the output layer.
        beta (NumPy 2d array): Weights parameter matrix from the hidden layer
            to the output layer.

    Returns:
        1d or 2d Numpy array: Returns the neural network output vector.
    """
    if x.ndim == 1:
        w = np.dot(x, alpha) + alpha_0
        z = sigma(w)
        t = np.dot(z, beta) + beta_0
        y_tilde = g(t)
        return y_tilde, t, z, w
    else:
        w = np.dot(x, alpha) + alpha_0
        z = sigma(w)
        t = np.dot(z, beta) + beta_0
        y_tilde = g(t)
        return y_tilde, t, z, w

def neural_network_gradient(x, y, sigma, sigma_prime,
                            alpha_0, alpha, beta_0, beta):
    """ Evaluates the gradient of the loss function of the single-layer neural
    network when the loss function is the cross-entropy and the output function
    is the softmax operator.

    Args:
        x (2d Numpy array): Observed input matrix x.
        x (1d Numpy array): Observed outcome vector y.
        sigma (function: array-like to array-like): Activation function (e.g.
            sigmoid function).
        sigma_prime (function: array-like to array-like): Derivative of sigma.
        alpha_0 (NumPy array): Bias parameter vector from the input layer to
            the hidden layer.
        alpha (NumPy 2d array): Weights parameter matrix from the input layer
            to the hidden layer.
        beta_0 (NumPy array): Bias parameter vector from the hidden layer to
            the output layer.
        beta (NumPy 2d array): Weights parameter matrix from the hidden layer
            to the output layer.

    Returns:
        2d Numpy array: Returns the Jacobian of the neural network output
            vector.
    """
    g = lambda x: softmax(x, slice(None))

    y_tilde, t, z, w = neural_network(x, g, sigma,
                                      alpha_0, alpha, beta_0, beta)
    delta = back_propagation_delta(y, y_tilde)
    s = back_propagation_s(delta, w, beta, sigma_prime)

    p, M = alpha.shape
    K = beta.shape[1]
    ret = np.zeros((p+1)*M + (M+1)*K)

    ret[:M] = np.sum(s, 0)
    ret[M:(p+1)*M] = np.dot(np.transpose(x), s).reshape(M * p, order='F')
    ret[(p+1)*M:(p+1)*M + K] = np.sum(delta, 0)
    ret[(p+1)*M + K:] = np.dot(np.transpose(z), delta
                               ).reshape(K * M, order='F')

    return ret

def cross_entropy(y, y_tilde):
    """ Computes the cross-entropy loss function for a given model.

    Args:
        y (1d or 2d Numpy array): Observed outcomes.
        y_tilde (1d or 2d Numpy array): Computed outcomes.

    Returns:
        float or 1d Numpy array: Returns the cross-entropy loss.
    """
    return -np.sum(y * np.log(y_tilde))

def theta_map(theta, p, M, K):
    """ Auxiliary function used in the `calibrate_neural_network function. It
    splits the 1d parameter array into the bias vectors and weight matrices.
    """
    alpha_0_slice = slice(M)
    alpha_slice = slice(M, M*(p+1))
    beta_0_slice = slice(M*(p+1), M*(p+1) + K)
    beta_slice = slice(M*(p+1) + K, None)
    return (theta[alpha_0_slice], theta[alpha_slice].reshape((p,M), order='F'),
            theta[beta_0_slice], theta[beta_slice].reshape((M,K), order='F'))

def calibrate_neural_network(K, M, p, sigma, sigma_prime, y, x,
                             gradient_descent_method, **kwargs):
    """ Calibrates a neural network with one hidden layer for classification.

    Args:
        K (int): Number of classes for classification
        M (int): Number of neurons in the hidden layer
        p (int): Number of features in `x`
        sigma (function: array-like to array-like): Activation function (e.g.
            sigmoid function).
        sigma_prime (function: array-like to array-like): Derivative of sigma.
        y (1d NumPy array): Y observations in the training set
        x (2d NumPy array): X observations in the training set
        gradient_descent_method (string): Selects the gradient descent method.
            The available methods are: "Gradient Descent" and
            "Mini-batch Gradient Descent".
        **kwargs: arguments passed to the optimiser

    Returns:
        array-like: returns the approximated value of x such that f(x) = 0
            given the algorithm interruption conditions.
    """

    y_ = np.zeros((len(y), 10))
    for i in range(10):
        y_[y == i, i] = 1
    y = y_

    #Setting up the initial guess
    x0 = np.zeros((p+1)*M + (M+1)*K)
    alpha = np.eye(p, M)
    beta = np.eye(M, K)
    x0[M:(p+1)*M] = alpha.reshape(M * p, order='F')
    x0[(p+1)*M + K:] = beta.reshape(K * M, order='F')

    #Setting up the gradient functions
    gradient_function = (lambda theta:
        neural_network_gradient(x, y, sigma, sigma_prime,
                                *theta_map(theta, p, M, K)))
    sampled_gradient_function = (lambda theta, sample_bool:
        neural_network_gradient(x[sample_bool, :], y[sample_bool], sigma,
                                sigma_prime, *theta_map(theta, p, M, K)))

    # Performing the optimisation
    if gradient_descent_method == "Gradient Descent":
        return gradient_descent(x0, gradient_function, **kwargs)
    if gradient_descent_method == "Mini-batch Gradient Descent":
        return mini_batch_gradient_descent(x0, x.shape[0],
                                           sampled_gradient_function, **kwargs)
    return None

"""
This exercise demonstrates the neural network model for the MNIST data set.

Four model settings are calibrated. Then, the time elapsed and accuracy of
these setting are computed.
"""

import datetime as dt
import numpy as np
import pandas as pd
import cPickle as pk
from os.path import isfile
from timeit import default_timer as timer
from mnist_tools import download_mnist_file, load_mnist
from neural_network import (argmax_classifier, neural_network, theta_map,
                            softmax, sigmoid, derivative_of_sigmoid,
                            calibrate_neural_network)

def initialise_data():
    """ Downloads MNIST data set, if needed, and returns a four-element tuple
    which contains its training set for y and x, respectively, and the
    dimensions of y and x, respectively.
    """
    print "Downloading data"
    download_mnist_file()
    
    print "Loading data"
    train_set, _, _ = load_mnist()

    y = train_set[1]
    x = train_set[0]

    K = 10
    p = x.shape[1]

    return y, x, K, p

def calibrate_model_1():
    """ Calibrate a neural network model where the activation function is the
    sigmoid function, the output function is the softmax function, the hidden
    layer has 10 neurons and its trained for 300 iterations.
    """
    y, x, K, p = initialise_data()

    sigma = sigmoid
    sigma_prime = derivative_of_sigmoid
    M = 10

    print "Calibrating model 1 (M = 10)"

    start_t = timer()
    theta = calibrate_neural_network(K, M, p, sigma, sigma_prime, y, x,
                                     "Mini-batch Gradient Descent")
    end_t = timer()
    with open('theta10.pickle', 'wb') as f:
        pk.dump(theta, f)
    with open('time10.pickle', 'wb') as f:
        pk.dump(end_t - start_t, f)

def calibrate_model_2():
    """ Calibrate a neural network model where the activation function is the
    sigmoid function, the output function is the softmax function, the hidden
    layer has 50 neurons and its trained for 300 iterations.
    """
    y, x, K, p = initialise_data()

    sigma = sigmoid
    sigma_prime = derivative_of_sigmoid
    M = 50

    print "Calibrating model 2 (M = 50)"

    start_t = timer()
    theta = calibrate_neural_network(K, M, p, sigma, sigma_prime, y, x,
                                     "Mini-batch Gradient Descent")
    end_t = timer()
    with open('theta50.pickle', 'wb') as f:
        pk.dump(theta, f)
    with open('time50.pickle', 'wb') as f:
        pk.dump(end_t - start_t, f)

def calibrate_model_3():
    """ Calibrate a neural network model where the activation function is the
    sigmoid function, the output function is the softmax function, the hidden
    layer has 150 neurons and its trained for 300 iterations.
    """
    y, x, K, p = initialise_data()

    sigma = sigmoid
    sigma_prime = derivative_of_sigmoid
    M = 150

    print "Calibrating model 3 (M = 150)"

    start_t = timer()
    theta = calibrate_neural_network(K, M, p, sigma, sigma_prime, y, x,
                                     "Mini-batch Gradient Descent")
    end_t = timer()
    with open('theta150.pickle', 'wb') as f:
        pk.dump(theta, f)
    with open('time150.pickle', 'wb') as f:
        pk.dump(end_t - start_t, f)

def calibrate_model_4():
    """ Calibrate a neural network model where the activation function is the
    sigmoid function, the output function is the softmax function, the hidden
    layer has 10 neurons and its trained for 1500 iterations.
    """
    y, x, K, p = initialise_data()

    sigma = sigmoid
    sigma_prime = derivative_of_sigmoid
    M = 10

    print "Calibrating model 4 (M = 10, 1500 iters)"

    start_t = timer()
    theta = calibrate_neural_network(K, M, p, sigma, sigma_prime, y, x,
                                     "Mini-batch Gradient Descent",
                                     max_iterations=1500)
    end_t = timer()
    with open('theta10_2.pickle', 'wb') as f:
        pk.dump(theta, f)
    with open('time10_2.pickle', 'wb') as f:
        pk.dump(end_t - start_t, f)

def exercise():
    """ Calibrate the four models above, if that has not yet been done, and
    print the time elapsed to train the models with their accuracy rates for
    the training, validation and test sets.
    """
    if not isfile('theta10.pickle'):
        calibrate_model_1()
    if not isfile('theta50.pickle'):
        calibrate_model_2()
    if not isfile('theta150.pickle'):
        calibrate_model_3()
    if not isfile('theta10_2.pickle'):
        calibrate_model_4()

    print "Loading data"

    train_set, valid_set, test_set = load_mnist()
    y = train_set[1]
    x = train_set[0]
    y_valid = valid_set[1]
    x_valid = valid_set[0]
    y_test = test_set[1]
    x_test = test_set[0]

    K = 10
    p = x.shape[1]

    print "Computing the results"

    with open('theta10.pickle', 'rb') as f:
        theta_10 = pk.load(f)
    with open('time10.pickle', 'rb') as f:
        time_elapsed_10 = pk.load(f)
    with open('theta50.pickle', 'rb') as f:
        theta_50 = pk.load(f)
    with open('time50.pickle', 'rb') as f:
        time_elapsed_50 = pk.load(f)
    with open('theta150.pickle', 'rb') as f:
        theta_150 = pk.load(f)
    with open('time150.pickle', 'rb') as f:
        time_elapsed_150 = pk.load(f)
    with open('theta10_2.pickle', 'rb') as f:
        theta_10_2 = pk.load(f)
    with open('time10_2.pickle', 'rb') as f:
        time_elapsed_10_2 = pk.load(f)

    models = [('M = 10', 10, theta_10, time_elapsed_10),
              ('M = 50', 50, theta_50, time_elapsed_50),
              ('M = 150', 150, theta_150, time_elapsed_150),
              ('M = 10 (1500 iters)', 10, theta_10_2, time_elapsed_10_2)]
    
    index = pd.Index([model[0] for model in models], name='Number of neurons')
    results = pd.DataFrame([dt.timedelta(seconds=model[3]) for model in models],
                           index=index, columns=['Time elapsed'])

    sigma = sigmoid
    g = lambda x: softmax(x, slice(None))
    accuracy = lambda y, predictor, x, M, theta: (
        np.sum(y==predictor(x, M, theta))/float(len(y)))
    predictor = lambda x, M, theta: (
        argmax_classifier(neural_network(x, g, sigma,
                                         *theta_map(theta, p, M, K))[0]))

    results["Accuracy (training)"] = [
        accuracy(y, predictor, x, model[1], model[2]
                 ) for model in models]
    results["Accuracy (validation)"] = [
        accuracy(y_valid, predictor, x_valid, model[1], model[2]
                 ) for model in models]
    results["Accuracy (test)"] = [
        accuracy(y_test, predictor, x_test, model[1], model[2]
                 ) for model in models]

    results = results.T
    results.to_csv('results.csv')

    print results

if __name__ == "__main__":
    exercise()
"""
Functions for downloading the MNIST data set, loading into memory and showing
characters.

"""

import cPickle as pk
import gzip as gz
import numpy as np
from pylab import imshow, cm
from requests import get
from os.path import isfile

deeplearning_file_name = "mnist.pkl.gz"
deeplearning_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

def download_mnist_file(url=deeplearning_url,
                        file_name=deeplearning_file_name):
    """ Downloads MNIST dataset from the specified `url` as saves it with the
    specified `file_name`.
    """

    if isfile(file_name):
        print ("The MNIST file has been found in the current directory. "
               "Downloading is not needed.")
        return

    with open(file_name, "wb") as f:
        f.write(get(url).content)

def load_mnist(file_name=deeplearning_file_name):
    """ Loads MNIST dataset pickle from `file_name` and returns its contents.
    """
    with gz.open(file_name, "rb") as f:
        return pk.load(f)

def show_mnist_char(array):
    """ Plots an MNIST character based on 1D numpy `array` with 784 entries."""
    imshow(np.reshape(array, (28,28)), cmap=cm.gray)
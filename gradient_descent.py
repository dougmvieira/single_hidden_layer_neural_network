"""
Gradient descent and mini-batch gradient descent methods.

"""

import numpy as np


def gradient_descent(x0, gradient_function, alpha=1.e-4, max_iterations=1000,
                     verbose=True):
    """ Attempts to find x such that f(x) = 0 using the gradient descent
    method.
    
    Args:
        x0 (array-like): Initial guess for x.
        gradient_function (function: array-like to array-like): Function that
            computes the gradient of the objective function.
        alpha (Optional[float]): scales the gradient descent step size.
        max_iterations (Optional[int]): Sets a limit for the number of
            iterations of the gradient descent algorithm.
        stop_function (Optional[function: array-like to bool]): User-defined
            function to interrupt the algorithm. The function is fed with the
            current value of x and if it returns `True`, the algorithm stops.

    Returns:
        array-like: returns the approximated value of x such that f(x) = 0
            given the algorithm interruption conditions.
    """
    x = x0
    for i in range(max_iterations):
        grad_ = gradient_function(x)
        step = -alpha * grad_
        print "grad_max =", np.max(grad_)
        x += step

    if verbose:
        print 'Maximum number of iterations (', max_iterations, ') reached.'

    return x

def mini_batch_gradient_descent(x0, m, sampled_gradient_function, m0=1000,
                                max_iterations=300, alpha=1e-3,
                                verbose=True):
    """ Attempts to find x such that f(x) = 0 using the mini-batch gradient
    descent method.
    
    Args:
        x0 (array-like): Initial guess for x.
        gradient_function (function: array-like to array-like): Function that
            computes the gradient of the objective function.
        m0 (Optional[int]): mini-batch size
        alpha (Optional[float]): scales the mini-batch gradient descent step
            size.
        max_iterations (Optional[int]): Sets a limit for the number of
            iterations of the mini-batch gradient descent algorithm.
        stop_function (Optional[function: array-like to bool]): User-defined
            function to interrupt the algorithm. The function is fed with the
            current value of x and if it returns `True`, the algorithm stops.

    Returns:
        array-like: returns the approximated value of x such that f(x) = 0
            given the algorithm interruption conditions.
    """
    x_star = np.copy(x0)
    d = np.ceil(float(m)/m0)
    thresholds = np.expand_dims(np.arange(d)/d, axis=0)
    batch_bool = (np.expand_dims(np.random.rand(m), axis=1) >= thresholds)
    batch_bool[:, :-1] = np.logical_xor(batch_bool[:, :-1], batch_bool[:, 1:])
    for i in range(max_iterations):
        for j in range(int(d)):
            alpha_n = 1000*alpha/(i*d + j + 1000)
            x_star -= alpha_n * sampled_gradient_function(x_star,
                                                          batch_bool[:, j])

    if verbose:
        print 'Maximum number of iterations (', max_iterations, ') reached.'

    return x_star
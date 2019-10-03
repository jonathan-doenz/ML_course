# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx @ w
    return - tx.T @ e / N


def gradient_descent(
        y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
    
    Parameters
    ----------
    y: n-by-1 array
        target vector
    tx: n-by-d array
        data vector
    initial_w: d-by-1 array
        initial model vector
    max_iters: int
        number of iterations to perform
    gamma: float
        step size
    
    Returns 
    -------
    loss: float
        MSE error between target vector y and predicted points tx @ w
    w: d-by-1 array
        updated model
    """
    # model's weights' vector
    w = initial_w
    for n_iter in range(max_iters):
        # number of data points
        N = len(y)
        # compute stochastic gradient
        stoch_gradient = compute_stoch_gradient(y, tx, w)
        # compute error
        e = compute_error_vec(y, tx, w)
        # compute loss
        loss = e.T @ e / (2 * N)
        # update weights' vector
        w = w - gamma * stoch_gradient

    return loss, w

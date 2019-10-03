# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w)

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm.
    
    Parameters
    ----------
    y: n-by-1 array
        target vector
    tx: n-by-d array
        data vector
    batch_size: int
        number of batches to sample
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
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # number of data points
            N = len(minibatch_y)
            # compute stochastic gradient
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # compute error
            e = compute_error_vec(minibatch_y, minibatch_tx, w)
            # compute loss
            loss = e.T @ e / (2 * N)
            # update weights' vector
            w = w - gamma * stoch_gradient

    return loss, w



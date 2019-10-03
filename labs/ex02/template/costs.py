# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_error_vec(y, tx, w):
    """Compute the error vector: y - Xw"""
    return y - tx @ w

def compute_loss(y, tx, w, method="MSE"):
    """Calculate the loss using mean squared error (MSE)."""
    N = len(y)
    e = compute_error_vec(y, tx, w)
    if method == "MSE":
        loss = e.T @ e / N
    elif method == "MAE":
        loss = np.abs(e).T @ np.ones(N)
    
    return loss

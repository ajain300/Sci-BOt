########################################################
# Set of basic acquisition functions to be used in AL loop
########################################################

import numpy as np
from scipy.stats import norm

def Expected_Improvement(pred_y, pred_y_std, y_best, type = 'MAX'):
    """
    Expected Improvement
    """
    # utility function
    u = u_x(y_best, pred_y, pred_y_std, type = type)
    ei = (y_best - pred_y) * norm.cdf(u) + pred_y_std * norm.pdf(u)
    return ei

def u_x(y_best, pred_y, pred_y_std, type = 'MIN'):
    if type == 'MIN':
        return (y_best - pred_y) / pred_y_std
    elif type == 'MAX':
        return (pred_y - y_best) / pred_y_std

def Probability_of_Improvement(pred_y, pred_y_std, y_best):
    u = u_x(y_best, pred_y, pred_y_std, type = 'MIN')
    pi = norm.cdf(u)
    return pi

def Upper_Confidence_Bound(pred_y, pred_y_std, y_best):
    u = u_x(y_best, pred_y, pred_y_std, type = 'MIN')
    ucb = pred_y + 2 * pred_y_std * norm.pdf(u)
    return ucb

"""
Collection of functions specific to the project
"""
import numpy as np
import math

def corr2(a,b):
    # Correlation coefficient of N x N x T array
    a = a - np.mean(a)
    b = b - np.mean(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    
    return r

def RMSE(a,b):
    # Root mean square error (absolute error)
        
    a_2norm = np.linalg.norm(a.flatten(), ord=2)
    ab_2norm = np.linalg.norm(a.flatten()-b.flatten(), ord=2)
    
    RMSE = ab_2norm/a_2norm
    
    return RMSE



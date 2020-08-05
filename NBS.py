#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cvxpy as cp
import math


# In[23]:


# Takes 2-D agent value np.ndarray and 1-D outside option np.ndarray
# Returns Probability Distribution that satisfies Nash Bargaining Solution
def NBS(val_mat, O, caps):
    if type(val_mat) != np.ndarray:
        raise Exception("val_mat must be type np.ndarray. Current type: ", type(val_mat))
    if type(O) != np.ndarray:
        raise Exception("O must be type np.ndarray. Current type: ", type(O))
    if np.ndim(val_mat) != 2:
        raise Exception("val_mat must be a 2-D array. Current Shape: ", val_mat.shape)
    if np.ndim(O) != 1:
        raise Exception("O must be a 1-D array. Current Shape: ", 0)
        
    V = cp.Parameter(val_mat.   shape, nonneg=True)
    V.value = val_mat
    
    # create matrix of same shape holding probability variables
    P = cp.Variable(nonneg=True, shape=V.shape)
    U = cp.Variable(nonneg=True, shape=V.shape[0])

    # Setting Objective Function
    objective_fn = cp.sum(-cp.log(U-O))
    
    # Setting constraint that ∑_{j}p for all i == 1
    constraints = []
    for i in range(V.shape[0]):
        constraints.append(cp.sum(cp.multiply(P[i], V[i]))-U[i]==0.0)
        constraints.append(cp.sum(cp.multiply(P[i], V[i])) - O[i] >= 0.0)
        constraints.append(cp.sum(P[i])<=1.0)

    # Setting constraint that ∑_{i} p for all j == 1
    for j in range(V.shape[1]):
        col_sum = 0
        for i in range(V.shape[0]):
            col_sum += P[i][j]
        constraints.append(col_sum <= caps[j])

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve()
    
    if P.value is None:
        return P.value

    P_vals = np.array(P.value)

    return P_vals


# In[9]:





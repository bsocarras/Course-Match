#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cvxpy as cp
import math

# Takes 2-D agent value np.ndarray and 1-D outside option np.ndarray
# Returns Probability Distribution that satisfies Nash Bargaining Solution
def NBS(val_mat, num_courses, O, caps):
    if type(val_mat) != np.ndarray:
        raise Exception("val_mat must be type np.ndarray. Current type: ", type(val_mat))
#     if type(O_mat) != np.ndarray:
#         raise Exception("O must be type np.ndarray. Current type: ", type(O))
    if np.ndim(val_mat) != 2:
        raise Exception("val_mat must be a 2-D array. Current Shape: ", val_mat.shape)
#     if np.ndim(O_mat) != 1:
#         raise Exception("O must be a 1-D array. Current Shape: ", 0)
        
#     V = cp.Parameter(val_mat.shape, nonneg=True)
#     V.value = val_mat
    V = val_mat
    
    # create matrix of same shape holding probability variables
    P = cp.Variable(nonneg=True, shape=V.shape)

#     O = cp.Parameter(O_mat.shape, nonneg=True)
#     O.value = O_mat
    
#     num_courses = cp.Parameter(num_courses_mat.shape, nonneg=True)
#     num_courses.value = num_courses_mat
    
#     caps = cp.Parameter(caps_mat.shape, nonneg=True)
#     caps.value = caps_mat

    # Setting Objective Function
    objective_fn = cp.sum(-cp.log(cp.sum(cp.multiply(P, V), axis=1)-O))
    
    # Setting constraint that ∑_{j}p for all i < num_courses[i]
    constraints = []
    for i in range(V.shape[0]):
#         constraints.append(cp.sum(cp.multiply(P[i], V[i]))-U[i]>=0.0) # - eps from RHS
        constraints.append(cp.sum(cp.multiply(P[i], V[i])) - O[i]>=0.0) # - eps from RHS
        constraints.append(cp.sum(P[i])<=num_courses[i])

    # Setting constraint that ∑_{i}p for all j <= caps_{j}
    for j in range(V.shape[1]):
        col_sum = 0
        for i in range(V.shape[0]):
            col_sum += P[i][j]
            constraints.append(P[i][j] <= 1.)
        constraints.append(col_sum - caps[j] <= 0)

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(solver=cp.SCS)#, max_iters=50000)
    
    if P.value is None:
        return P.value

    P_vals = np.array(P.value)

    return P_vals





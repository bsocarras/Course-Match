#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import math
import sys

import NBS

np.set_printoptions(precision=9)


# In[2]:


# Takes np.ndarray of values for n agents
# Returns 2 np.ndarrays and list: values for selected agents, values for non-selected agents
# ciel(n/2) selected agents, floor(n/2) non-selected agents, indices of selected agents
def random_subset(vals, selected_indices):
    if type(vals) != np.ndarray:
        raise Exception("vals is not type np.ndarray, but type: ", type(vals))
    if type(selected_indices) != list:
        raise Exception("selected_indices is not type list, but type: ", type(selected_indices))
    
    num_selected = int(math.ceil(float((vals.shape[0] - len(selected_indices)) / 2)))
    new_indices = []
    while len(new_indices) != num_selected:
        r = random.randint(0, vals.shape[0]-1)
        if r not in selected_indices:
            selected_indices.append(r)
            new_indices.append(r)
    
    selected_agents = np.zeros(shape=(num_selected, vals.shape[1]))
    selected_index = 0    
    for i in range(vals.shape[0]):
        if (i in new_indices):
            selected_agents[selected_index] = vals[i]
            selected_index += 1
    return selected_agents, new_indices


# In[3]:


def PA(selected, cap):
    num_agents = selected.shape[0]
    deleted = []
    # Checking for indifferent agents
    for i in range(selected.shape[0]):
        avg = np.average(selected[i])
        avg_array = np.zeros(selected.shape[1])
        for j in range(selected.shape[1]):
            avg_array[j] = avg
        if np.allclose(selected[i], avg_array):
            deleted.append(i)
    
    selected = np.delete(selected, deleted, axis=0)
    
    if selected.shape[0] == 0:
        return np.zeros((num_agents, selected.shape[1]))
    else:
    
        # Calculating outside option
        O = np.zeros(selected.shape[0])
        for i in range(O.size):
            O[i] = np.sum(np.multiply(selected[i], cap)) / selected.shape[1]

        # Nash Social Welfare Optimal Probability Matrix
        nsw = NBS.NBS(selected, O, cap)
        if nsw is None:
            raise Exception("NBS returned None.")

        # NSW Optimal Utility
        util = np.sum(np.multiply(nsw, selected), axis=1)

        # Calculating f for each agent
        if nsw.shape[0] > 1:
            f = np.zeros(selected.shape[0])
            for i in range(f.size):
                new_sel = np.delete(selected, i, axis=0)
                new_O = np.delete(O, i)

                i_exclusive = NBS.NBS(new_sel, new_O, cap)
                if i_exclusive is None:
                    raise Exception("NBS returned None while calculating f_{",i,"}.")

                new_util = np.sum(np.multiply(new_sel, i_exclusive), axis=1)

                num = 1
                denom = 1
                for j in range(i):
                    num *= util[j]
                    denom *= new_util[j]
                if i < new_util.shape[0]:
                    for j in range(i+1, util.shape[0]):
                            num *= util[j]
                            denom *= new_util[j-1]

                f[i] = float(num/denom)
                if np.isclose(f[i], 1.0):
                    f[i] = 1.0

            # Applying f to each agent
            for i in range(nsw.shape[0]):
                nsw[i] *= f[i]


    
    # Adding back indifferent agents if necessary
    if len(deleted) == 0:
        probs = nsw
    else:
        j=0
        probs = np.zeros((num_agents, selected.shape[1]))
        next_index = 0
        while len(deleted) > 0:
            next_del = deleted.pop(0)
            for i in range(next_index, next_del):
                probs[i] = nsw[j]
                j += 1
            # probs already zeros #
            next_index = next_del+1
        for i in range(next_index, probs.shape[0]):
            probs[i] = nsw[j]
            j+=1
        
        
    return probs


# In[4]:


def pref_att(num_agents, num_items, p):
    array = np.zeros(shape=(num_agents, num_items))
    array[0] = np.random.rand(num_items)
    for i in range(1, num_agents):
        rint = np.random.randint(0, i)
        array[i] = array[rint]
        for j in range(1, num_items):
            r = np.random.rand()
            if r < p:
                array[i][j] = np.random.rand()
    return array


# In[12]:


# Takes 2-D np.ndarray vals matrix, 1-D np.ndarray course caps matrix, and int of agents left to stop recursing
# Returns 2-D np.ndarray Probability Distribution
def RPIk_recurse(vals, selected_indices, cap, k, n_knot):
    print("\n-----------------\n")
    print("cap: ", cap)
    # Algorithm Start:
    n_hat = vals.shape[0]-len(selected_indices)
    if n_hat <= n_knot:
        P = np.zeros(shape=vals.shape)
        uni_probs = cap / n_hat
        for i in range(P.shape[0]):
            if i not in selected_indices:
                P[i] = uni_probs
        return P
    else:
        # Seperate half of agents randomly
        selected, new_indices = random_subset(vals, selected_indices)
        new_indices.sort() # Need to sort for Tweaking Output and Putting Vals Back
        
        # Partial Allocation Mechanism
        P_selected = PA(selected, cap)
        
        # Tweaking PA Mechanism Output
        total_alloc = np.sum(P_selected, axis=1)
                    
        g = np.sum(np.multiply(P_selected, k[new_indices]), axis=1)
#         kq_sums = np.zeros(P_selected.shape[1]) ### TO BE REMOVED
        for i in range(P_selected.shape[0]):
            first_part = 1.0-g[i]
#             if np.isclose(first_part, 0.):
#                 first_part = 0

            for j in range(P_selected.shape[1]):
                second_part = cap[j]/n_hat
                P_selected[i][j] = float(P_selected[i][j] * k[new_indices[i]][j] + first_part*second_part)
#                 kq_sums[j] += P_selected[i][j] * k[new_indices[i]][j]
        
#         for j in range(P_selected.shape[1]):
#             if np.sum(g) > cap[j] - kq_sums[j]:### TO BE REMOVED
#                 raise Exception("We got em. g sum: ", np.sum(g), " | kq sum: ", kq_sums[j])

        # Recursively calling RPI_recurse
        cap = cap - np.sum(P_selected, axis=0)
#         for i in range(cap.size):
#             if np.isclose(cap[i], 0.):
#                 cap[i] = 0
        
        P = np.zeros(shape=vals.shape)
        
        # Putting Vals Back
        for i in range(len(new_indices)):
            P[new_indices[i]] = P_selected[i]
        return np.add(RPIk_recurse(vals, selected_indices, cap, k, n_knot),  P)


# In[13]:


# Takes 2-D np.ndarray value matrix and int lowest n
# Returns 2-D np.ndarray probability matrix
def RPIk(v, n_knot):
    #---------------------------------------------INVARIANT TESTS-------------------------------------------
    if type(v) != np.ndarray:
        raise Exception("v must be type np.ndarray. Current type: ", type(v))
    if type(n_knot) != int:
        raise Exception("n_knot must be type int. Current type: ", type(n_knot))
    
    if np.ndim(v) != 2:
        raise Exception("v must be a 2-D np.ndarray. Current shape: ", v.shape)
        
#     if n_knot < 4:
#         raise Exception("n_knot must be >= 4. Current n_knot: ", n_knot)
    #-----------------------------------------------TESTS END-----------------------------------------------
    
    # Making sure v has dimensions n x n
    num_agents = v.shape[0]
    num_items = v.shape[1]
    
    if num_agents > num_items: 
        z = np.zeros(shape=(num_agents, num_agents-num_items))
        v = np.concatenate((v, z), axis=1)
        
    elif num_items > num_agents:
        z = np.zeros(shape=(num_items-num_agents, num_items))
        v = np.concatenate((v, z), axis=0)
        
    
#     print("k:\n", k)
     
    return RPIk_recurse(v, [], np.ones(v.shape[1]), k, n_knot)


# In[7]:


def monte_carlo_RPIk(V, n):
    max_size = max(V.shape[0], V.shape[1])
    p = np.zeros(shape=(max_size, max_size))
    for i in range(n):
        p = np.add(p, RPIk(V, 4))
        
    ax_0 = []
    for i in range(V.shape[0], p.shape[0]):
        ax_0.append(i)
    p = np.delete(p, ax_0, 0)
    
    ax_1 = []
    for j in range(V.shape[1], p.shape[1]):
        ax_1.append(j)
    p = np.delete(p, ax_1, 1)
        
    return p/n


# In[8]:


# Returns expected value of agent/item utility matrix determined empirically
# Takes value matrix, optionally number of iterations (default 1000)
def monte_carlo_RSD(V, n):     
    p = np.zeros(V.shape)
    for i in range(n):
        p = np.add(p, RSD.RSD(V))
    return p/n


# In[9]:


# Takes num agents, num items, distribution for agent values to be drawn from, and optionally number of trials
# Returns worst-case ratio 
def worst_case_ratio(V, Func, n=1000):
    if not(type(V) == np.ndarray):
        raise Exception("V must be a numpy.ndarray, currently type: ", type(V))
    
    P_NBS = NBS.NBS(V,np.zeros(V.shape[0]), np.ones(V.shape[1]))
    if P_NBS is None: 
        raise Exception("NBS failed, returned None")
        
    U_NBS = np.sum(np.multiply(V, P_NBS), axis=1)
    U_Func = np.sum(Func(V,n), axis=1)
    
    U_ratio = np.divide(U_NBS, U_RDS)
    return np.amax(U_ratio)


# In[10]:


def i_mat(size):
    V = np.zeros((size, size))
    for i in range(size):
        V[i][i] = 1
    return V







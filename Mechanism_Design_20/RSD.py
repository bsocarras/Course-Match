#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import heapq 
import random


# In[86]:


# Takes an agent's values, returns index of most values item for this agent's values
# Invariant: v is a 1-D array with items not to be considered (already assigned) already excluded
def find_favored_item(v, taken_cp):
    max_val = 0
    max_index = None
    curr = 0
    while len(taken_cp) != 0:
        min_tcp = min(taken_cp)
        for i in range(curr, min_tcp):
            if v[i] >= max_val:
                max_index = i
                max_val = v[i]
            curr += 1
        taken_cp.remove(min_tcp)
        curr += 1
    for i in range(curr, v.size):
            if v[i] >= max_val:
                max_index = i
                max_val = v[i]
        
    return max_index


# In[74]:


# Takes number of agents, returns list of order that agents select items in
def rand_order_u01(num_agents):
    pq = []
    for i in range(num_agents):
        heapq.heappush(pq, (random.random(), i))
    pq.sort()
    order = []
    for pair in pq: 
        order.append(pair[1])
    return order


# In[90]:


# Random Serial Dictatorship algorithm
# V is an NxM numpy.ndarray detailing N agent's values for M items, returns array mapping items to agents
def RSD(V):
    # Check that V is an numpy.ndarray
    if type(V) is not np.ndarray:
        raise Exception("V is not a numpy.ndarray")

    # Assignments that map items (A indices) to agents (V row indices stored in A)
    A = np.zeros(V.shape)

    taken = [] # list of items that have been assigned

    # Actual RSD Algorithm
    order = rand_order_u01(V.shape[0])
    for o in order: 
        f_item = find_favored_item(V[o], taken.copy())
        if f_item is None:
            break
        taken.append(f_item)
        A[o][f_item] = 1

    return A


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:31:57 2018

@author: ekele
"""

import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:, 1], Y)
plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
Yhat = np.dot(X,w)

# plot it all together
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]),  sorted(Yhat))
plt.show()

# calculate r2
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1)/d2.dot(d2)

print("weights: ", w)
print("r-squared: ", r2)

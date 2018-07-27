#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:04:12 2018

@author: ekele
"""


import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

denominator = X.dot(X) - X.mean()*X.sum()
a = (X.dot(Y) - Y.mean()*X.sum()) / denominator

b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator


Yhat = a*X + b

plt.scatter(X,Y)
plt.plot(X,Yhat,color='r')
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1)/d2.dot(d2)

print('The r-squared score is: ', r2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:10:57 2017

@author: virajdeshwal
"""
'''
The main purpose of this project is to predict whether the new employee is saying 
truth about his previous salary or he is bluffing.
If the salary said by the new employee mathces with the curve fitting with the minimal error. Then he is saying truth.
Or else the employee is bluffing about the salary.'''

import pandas as pd
file = pd.read_csv('Position_Salaries.csv')
X = file.iloc[:,1:2].values
y = file.iloc[:, 2].values

#from sklearn.cross_validation import train_test_split
'''we do not need train and test split as our data set is pretty small.'''


#first lets do it with the linear regression

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X,y)


#lets fit the polynomial regression to the data 
from sklearn.preprocessing import PolynomialFeatures

poly= PolynomialFeatures(degree =3)

X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)


#lets fit the poly into linear regression .

linear_poly = LinearRegression()
linear_poly.fit(X_poly,y)

import matplotlib.pyplot as plt

plt.scatter(X,y, color='red')
plt.plot(X,linear.predict(X), color='blue')
plt.title('linear')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()
#now lets plot the polynomial degree plot.
plt.scatter(X,y, color='red')
plt.plot(X,linear_poly.predict(poly.fit_transform(X)), color='blue')
'''Do not fell in trap of taking X_poly for predict. We have to take fit_transform.
so that we can use that for any new variable of X'''
plt.title('polynomial')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()


'''for more natural curve. we can use grid function()'''
import matplotlib.pyplot as plt
import numpy as np
X_grid =np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid,linear_poly.predict(poly.fit_transform(X_grid)), color='blue')
plt.title('polynomial with natural plotting')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()













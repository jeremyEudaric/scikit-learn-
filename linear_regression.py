# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:46:06 2020

@author: jejeu
"""


import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
#import matplotlib.pyplot as plt



########################### preprocessing ##################################
iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)


x = iris.data[:,[2,3]]
b= np.shape(x)
print(b)

y= iris.target
#a= np.shape(y)
#print(a)





x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.3,random_state = 1, stratify=y)

# using train_test_split, we randomly split the x and y into 30 percent test data and 70 percent training data 


scaler = StandardScaler()

scaler.fit(x_train)
x_train_transforme = scaler.transform(x_train)
x_test_transforme = scaler.transform(x_test)



######################## Learning ######################################

linear = linear_model.LinearRegression()


#for train the model (learning)
linear.fit(x_train_transforme ,y_train)



###################### Evaluation ####################################

# evaluation of the model

model = linear.score(x_test_transforme, y_test)

print(model)

#################### Prediction ###################################

predictions = linear.predict(x_test_transforme)
print(predictions)


#To retrieve the intercept:
#print(linear.intercept_)
#For retrieving the slope:
#print(linear.coef_)



#plt.scatter(x_test, y_test,  color='gray')

#plt.show()

#Visualization of the training set results
#plt.scatter(x_test, y_test, color = 'red')
#plt.plot(x_test, linear.predict(x_train), color = 'green')
#plt.title('')
#plt.xlabel('')
#plt.ylabel('')
#plt.show()
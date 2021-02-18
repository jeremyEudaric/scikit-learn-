# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:57:33 2020

@author: jejeu
"""


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import datasets

diabete = datasets.load_wine()
x= diabete.data[:,[2,3]]
y = diabete.target
print(x,y)

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.3,random_state = 1, stratify=y)

scaler = StandardScaler()

scaler.fit(x_train) 

x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

###################################### learning ###########################

linear = linear_model.LinearRegression()

linear.fit(x_train_std, y_train)

####################################### evaluation ###################

model = linear.score(x_test_std, y_test)

print(model)

######################################## presiction#################

prediction = linear.predict(x_test_std)

print(prediction)



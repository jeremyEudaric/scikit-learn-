# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:28:25 2020

@author: jejeu
"""
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
 
# Dataset for cancer 
cancer = datasets.load_breast_cancer()

# We will print the target and feature for this dataset 
print(cancer.feature_names)
print(cancer.target_names)


x = cancer.data

y= cancer.target


x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.3,random_state = 1, stratify=y)

# using train_test_split, we randomly split the x and y into 30 percent test data and 70 percent training data 
# for  stratification , training and test subsets that have the same proportions of class labels as the input dataset
print(x_train)

print(y_train)

# Zero reprents malignant and  one begnin 
classes = ["malignant", "bengin"]
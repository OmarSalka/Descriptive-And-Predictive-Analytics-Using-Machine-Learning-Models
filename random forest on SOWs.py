# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:54:11 2018

@author: I870266
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
random_forest_dataset = pd.read_csv("linear_separability_assumed before clustering.csv")
#----------------------------------------------------------------------------------------------#
#--------------------Using the SOW characteristics as the independant variables----------------#
#----------------------------------------------------------------------------------------------#

x1 = random_forest_dataset.iloc[:, [23, 24, 25, 26, 27, 28, 29]].values
y1 = random_forest_dataset.iloc[:, 34].values

#assigning train and test sets
from sklearn.cross_validation import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.25, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)

#Fitting random forest classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state =0)#playing around with the #of trees can help you detet overfitting
classifier.fit(x1_train, y1_train)

#Predicting the test set results
y1_pred = classifier.predict(x1_test)

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1_test, y1_pred)
#----------------------------------------------------------------------------------------------#
#--------------------Using some SOW characteristics as the independant variables----------------#
#----------------------------------------------------------------------------------------------#

x1 = random_forest_dataset.iloc[:, [30, 31, 32, 33]].values
y1 = random_forest_dataset.iloc[:, 34].values

#assigning train and test sets
from sklearn.cross_validation import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.25, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)

#Fitting random forest classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state =0)#playing around with the #of trees can help you detet overfitting
classifier.fit(x1_train, y1_train)

#Predicting the test set results
y1_pred = classifier.predict(x1_test)

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1_test, y1_pred)
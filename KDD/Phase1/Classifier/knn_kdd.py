
# Iman Rezaei
# KNN classifier for KDD term project - Phase 1
# knn_kdd.py
# 4/8/16


import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing, datasets


# **************************************************************************

# import data file
dataFile = pd.read_csv("traindata.csv", header=0)

# CFA is our Y
cfa = dataFile['Correct First Attempt'].ravel(order='C')

# Student ID and Step ID will be our X   (input for prediction)
studentId = dataFile['Anon Student Id'].ravel(order='C')
stepName = dataFile['Step Name'].ravel(order='C')

# **************************************************************************

# Generate label encoders
le_studentId = preprocessing.LabelEncoder()
le_stepName = preprocessing.LabelEncoder()

# Encode student id and step name values
encoded_studentId = le_studentId.fit_transform(studentId)
encoded_stepName = le_stepName.fit_transform(stepName)

# **************************************************************************

# TODO:
# create dictionaries for encoded_studentId and encoded_stepName with keys being studentId & stepName
#  http://stackoverflow.com/questions/209840/map-two-lists-into-a-dictionary-in-python

# **************************************************************************

# TODO:
# we need a matrix like this for our X in fit():
# [ [6  103432]
#   [6  162908]
#   [6  93298]
#    ... ]
# each row consists of an element from encoded_studentId and one element from encoded_stepName.
# dimensions = 809694 x 2

# **************************************************************************

# Testing:
print("CFA: ", cfa)
print("encoded_stepName: ", encoded_stepName)
print("encoded_studentId:, ", encoded_studentId)
print(le_studentId.inverse_transform([encoded_studentId[2]]))

# **************************************************************************

# TODO:
# comment out x / delete after creating the right X.
# create new X based of the encoded values
X = [encoded_studentId, encoded_stepName]

# **************************************************************************

# Testing:
print(len(encoded_studentId))
print(len(encoded_stepName))

# X = np.matrix([encoded_studentId, encoded_stepName])
X = np.matrix(X)

# xtest = np.matrix(encoded_stepName)
# print("xtest: ", xtest.reshape(len(xtest), 1))

# **************************************************************************

# set Y (labels) to be our cfa
Y = cfa

# **************************************************************************

print("X: ", X)
print("X Shape: ", X.shape)
print("Y: ", Y)
print("Y Shape: ", Y.shape)

# Sample dataset to do experiments with:
# iris = datasets.load_iris()
# print("iris target: ", iris.target)
# print("iris data: ", iris.data)

yo = dataFile[['Anon Student Id', 'Step Name']]

# **************************************************************************

# KNN function:

# TODO:
# define a func as a wrapper for KNN.
# @param studentId, stepName
# @return prediction value

# looks up the keys for input from dict
# passes the values to KNN function

# knn = neighbors.KNeighborsClassifier()
# knn.fit(X, Y)

# print(knn.predict_proba([[6][103432]]))

# **************************************************************************




# Iman Rezaei
# KNN classifier for KDD term project - Phase 1
# knn_kdd.py
# 4/8/16


import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error
import math

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

# create dictionaries for encoded_studentId and encoded_stepName with keys being studentId & stepName


studentId_dict = dict(zip(studentId, encoded_studentId))
stepName_dict = dict(zip(stepName, encoded_stepName))

# print("studentId_dict: ", studentId_dict)
# print("stepName_dict: ", stepName_dict)


# **************************************************************************


# we need a matrix like this for our X in fit():
# [ [6  103432]
#   [6  162908]
#   [6  93298]
#    ... ]
# each row consists of an element from encoded_studentId and one element from encoded_stepName.
# dimensions = 809694 x 2


print("esn1: ", encoded_stepName[0])

x_array = []

for i in range(len(encoded_studentId)):
    x_array.append([encoded_studentId[i], encoded_stepName[i]])

# Convert x_array to np matrix
X = np.matrix(x_array)

# set Y (labels) to be our cfa
Y = cfa

print("shape of X: ", X.shape)


# **************************************************************************

# Testing:
print("CFA: ", cfa)
print("encoded_stepName: ", encoded_stepName)
print("encoded_studentId:, ", encoded_studentId)
print(le_studentId.inverse_transform([encoded_studentId[2]]))

# **************************************************************************


# Testing:
print(len(encoded_studentId))
print(len(encoded_stepName))


# **************************************************************************

print("X: ", X)
print("X Shape: ", X.shape)
print("Y: ", Y)
print("Y Shape: ", Y.shape)

# Sample dataset to do experiments with:
# iris = datasets.load_iris()
# print("iris target: ", iris.target)
# print("iris data: ", iris.data)


# **************************************************************************

# KNN function:


# define a func as a wrapper for KNN.
# @param studentId, stepName
# @return prediction value

# looks up the keys for input from dict
# passes the values to KNN function

knn = neighbors.KNeighborsClassifier(n_neighbors=41)
knn.fit(X, Y)


def predict_cfa(studentId, stepName):
    encoded_student_id = studentId_dict.get(studentId)
    print("encoded_student_id: ", encoded_student_id)
    encoded_step_name = stepName_dict.get(stepName)
    print("encoded_step_name: ", encoded_step_name)
    prediction = knn.predict([[encoded_student_id, encoded_step_name]])
    print("prediction: ", prediction[0])




# Testing:
# predict_cfa("0BrbPbwCMz", "3(x+2) = 15")

# **************************************************************************


# Treat the whole traindata as testdata and calculate RMSE:

def calculate_rmse():

    prediction_array = np.array([])
    for element in range(len(encoded_studentId)):
        prediction = knn.predict([[encoded_studentId[element], encoded_stepName[element]]])
        prediction_array = np.append(prediction_array, prediction[0])
        print("Prediction for step {:d} = {:d}".format(element, prediction[0]))

    print("prediction_array: ", prediction_array)
    rmse = math.sqrt(mean_squared_error(Y, prediction_array))
    print("RMSE = ", rmse)



calculate_rmse()


# **************************************************************************









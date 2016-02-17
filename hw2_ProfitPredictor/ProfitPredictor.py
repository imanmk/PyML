
# Iman Rezaei
# Professor Ding
# Machine Learning
# HW2
# 2/15/16


import time
from pylab import *
from numpy import *


# Data file name: "hwk2data.txt"


# ****************************************************************************
# Gradient Descent Function
# ****************************************************************************

def gradientDescent(x, y, alpha, convergenceError=0.00001, maxIter=5000):
    converged = False
    numIter = 0

    theta0 = random.choice(inputData[:, 0])  # initial random theta0
    # print("theta0 is: ", theta0)
    theta1 = random.choice(inputData[:, 1])  # initial random theta1
    # print("theta1 is: ", theta1)

    m = inputData[:, 1].size  # number of training data
    # print("number of training data is: ", m)
    # Cost function J (error)
    J = (1 / (2 * m)) * sum((theta0 + (theta1 * x[i]) - y[i]) ** 2 for i in range(m))

    # Repeat until convergence or until maxIter is reached

    # Turn on interactive mode
    # ion()

    while not converged:

        # Simultaneous update

        temp0 = theta0 - (alpha / m) * sum((theta0 + (theta1 * x[i]) - y[i]) for i in range(m))
        temp1 = theta1 - (alpha / m) * sum(((theta0 + (theta1 * x[i]) - y[i]) * x[i]) for i in range(m))

        theta0 = temp0
        theta1 = temp1

        currentJ = (1 / (2 * m)) * sum((theta0 + (theta1 * x[i]) - y[i]) ** 2 for i in range(m))
        # print("currentJ is: ", currentJ)
        # print("J is: ", J)
        if abs(J - currentJ) <= convergenceError:
            converged = True
            print("J is converged after ", numIter, " iterations.")
            print("theta0 is: ", theta0)
            print("theta1 is: ", theta1)

        # Plot cost function vs. number of iterations

        # scatter(numIter, J)
        # title("Cost Function VS. Number of Iterations")
        # xlabel("Number of Iterations")
        # ylabel("J()")

        # Pause quickly so GUI (plot) can get updated

        # pause(0.1 ** 10)
        # time.sleep(0.1 ** 10)

        # Update cost function (error) and number of iterations
        J = currentJ
        numIter += 1

        if numIter == maxIter:
            print("We have reached the MAX number of iterations! (%d)" % numIter)
            print("theta0 is: ", theta0)
            print("theta1 is: ", theta1)
            converged = True

    return theta0, theta1


# Function to predict profit with respect to the population of its city. (Gradient Descent)

def predictProfit(population):
    # Learning rate:
    alpha = 0.01
    theta0, theta1 = gradientDescent(inputData[:, 0], inputData[:, 1], alpha)
    profit = theta0 + (theta1 * (int(population) / (10 ** 5)))

    # Plot input data points
    scatter(inputData[:, 0], inputData[:, 1], marker='o', c='b', label='Training data')
    title("Profit With Respect to Population")
    xlabel("Population")
    ylabel("Profit")

    # Plot the linear regression
    plot(inputData[:, 0], theta0 + theta1 * inputData[:, 0], c='r', label='Current hypothesis')
    legend(loc=4)

    return profit


# ****************************************************************************
# Normal Equation Function
# ****************************************************************************

def normalEquation(x, y):
    onesArray = ones((len(x), 1), dtype=int)

    # Add ones for the first column of features (x0)
    featuresData = column_stack((onesArray, x))

    # Calculate theta based on the equation:
    # theta = inverse of (transpose of (X) * X) * transpose of (X) * y

    xtxinv = pinv(dot(transpose(featuresData), featuresData))
    xty = dot(transpose(featuresData), y)
    theta = dot(xtxinv, xty)

    return theta


# Function to predict profit with respect to the population of its city. (Normal Equation)

def predictProfitNormal(population):
    # Not necessary here because we have only one feature but if we had more features this would be the
    # vector of input features:
    inputFeatures = append(1, (int(population) / (10 ** 5)))

    theta = normalEquation(inputData[:, 0], inputData[:, 1])
    print(theta)
    profit = sum((theta[i] * inputFeatures[i]) for i in range(len(inputFeatures)))

    return profit


# ****************************************************************************
# Main
# ****************************************************************************

if __name__ == '__main__':

    # Get input data from file

    fileName = input("Please enter the name of your CSV file:\n")
    if fileName == "":
        inputData = loadtxt("hwk2data.txt", delimiter=',')
    else:
        inputData = loadtxt(fileName, delimiter=',')


    # Ask user to enter the population to get the profit:
    def prompt():

        print("\nPlease enter the number corresponding to your desired method of prediction")
        predictionMethod = input("Gradient Descent: 1     Normal Equation: 2\n\n")
        population = input("\nEnter population of a city to predict the profit of a coffee shop in that city:\n")

        if population.isdigit():
            if predictionMethod == "1":

                profit = predictProfit(population)
                print("The predicted profit is:  ", profit)
                show()
                prompt()
            elif predictionMethod == "2":
                profit = predictProfitNormal(population)
                print("The predicted profit is:  ", profit)
                prompt()
            else:
                print("Oops! The number entered does not match any methods.")
                prompt()
        else:
            print("Oops... Population must contain only digits!")
            prompt()


    prompt()

# ****************************************************************************

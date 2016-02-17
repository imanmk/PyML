# Iman Rezaei
# Professor Ding
# Machine Learning
# HW2
# 2/15/16


import time
from pylab import *
from numpy import *

# ****************************************************************************
# Get the data and plot it
# ****************************************************************************

# Get input data from file
inputData = loadtxt("hwk2data.txt", delimiter=',')
print(inputData[:, 0])

# Plot scatter graph from data
# scatter(inputData[:, 0], inputData[:, 1], marker='o', c='b')
# title("Profit With Respect to Population")
# xlabel("Population")
# ylabel("Profit")
# m = inputData[:, 1].size  # number of training data
# print("number of training data is: ", m)
# show()


# ****************************************************************************
# Gradient Descent Function
# ****************************************************************************

def gradientDescent(x, y, alpha, convergenceError=0.00001, maxIter=100000):
    converged = False
    numIter = 0

    theta0 = random.choice(inputData[:, 0])  # initial random theta0
    print("theta0 is: ", theta0)
    theta1 = random.choice(inputData[:, 1])  # initial random theta1
    print("theta1 is: ", theta1)

    m = inputData[:, 1].size  # number of training data
    print("number of training data is: ", m)
    # Cost function J (error)
    J = (1 / (2 * m)) * sum((theta0 + (theta1 * x[i]) - y[i]) ** 2 for i in range(m))

    # Repeat until convergence or until maxIter is reached

    # Turn on interactive mode
    #ion()

    while not converged:

        # Simultaneous update

        temp0 = theta0 - (alpha / m) * sum((theta0 + (theta1 * x[i]) - y[i]) for i in range(m))
        temp1 = theta1 - (alpha / m) * sum(((theta0 + (theta1 * x[i]) - y[i]) * x[i]) for i in range(m))

        theta0 = temp0
        theta1 = temp1

        currentJ = (1 / (2 * m)) * sum((theta0 + (theta1 * x[i]) - y[i]) ** 2 for i in range(m))
        print("currentJ is: ", currentJ)
        print("J is: ", J)
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


# ****************************************************************************
# Main - Plot the data with linear regression & Prompt
# ****************************************************************************

if __name__ == '__main__':



    # Learning rate:
    alpha = 0.001

    theta0, theta1 = gradientDescent(inputData[:, 0], inputData[:, 1], alpha)
    print(theta0)
    print(theta1)

    # Plot input data points
    scatter(inputData[:, 0], inputData[:, 1], marker='o', c='b', label='Training data')
    title("Profit With Respect to Population")
    xlabel("Population")
    ylabel("Profit")

    # Plot the linear regression
    plot(inputData[:, 0], theta0 + theta1 * inputData[:, 0], c='r', label='Current hypothesis')
    legend(loc=4)

    show()

    # Function to predict profit with respect to the population of its city
    def predictProfit(population):
        profit = theta0 + (theta1 * (int(population) / (10 ** 5)))
        return profit

    # Ask user to enter the population to get the profit:
    def prompt():
        population = input("\nEnter population of a city to predict the profit of a coffee shop in that city:\n")
        if population.isdigit():
            profit = predictProfit(population)
            print("The predicted profit is:  ", profit)
            prompt()
        else:
            print("Oops... Population must contain only digits!")
            prompt()

    prompt()

# ****************************************************************************



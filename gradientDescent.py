# import numpy as np
# import pandas as pd
# import matplotlib.pylab as plt
# plt.style.use(['ggplot'])
#
# # slope equation y = b + mx
#
# b = 4
# m = 3
#
# #X = 2 * np.random.rand(100, 1)
# #y = b+m*X + np.random.randn(100, 1)
#
#
# dataset = pd.read_csv('data.csv')
# X = dataset.iloc[:,0].values
# y = dataset.iloc[:,1].values
#
# # plot our data with reltion to x and y
# plt.plot(X, y, 'b.')
# plt.xlabel("$X$", fontsize=18)
# plt.ylabel("$Y$", rotation=0, fontsize=18)
# _ = plt.axis([0, 2, 0, 15])
#
#
#
# # Calculate cost function or MSE
# def cal_cost(theta, X, y):
#
# 	m = len(y)
#
# 	predictions = X.dot(theta)
# 	cost = (1/2*m) * np.sum(np.square(predictions - y))
#
# 	return cost
#
# def gradient_descent(X, y, theta, learning_rate, iterations):
# 	m = len(y)
# 	cost_history = np.zeros(iterations)
# 	theta_history   = np.zeros((iterations, 2))
#
# 	for it in range(iterations):
# 		prediction = np.dot(X, theta)
# 		theta = theta - learning_rate * (1/m) *(X.T.dot((prediction - y)))
# 		theta_history[it, :] = theta.T
# 		cost_history[it] = cal_cost(theta, X, y)
# 	return theta, cost_history, theta_history
#
#
# # Let's start with 1000 iterations and a learning rate of 0.01
# learningRate  = 0.01
# no_iterations = 1000
# # start with theta from gaussian distribution
# theta = np.random.randn(2, 1)
#
# X_b = np.c_[np.ones((len(X), 1)), X]
# theta, cost_history, theta_history = gradient_descent(X_b, y, theta, learningRate, no_iterations)
#
# print('Theta0 (predicted) (b):          {:0.3f},\nTheta1 (predicted) (m):          {:0.3f}'.format(theta[0][0],theta[1][0]))
# print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
#
#
# # Let's plot the cost history over iterations
# fig, ax = plt.subplots(figsize=(12, 8))
#
# ax.set_ylabel('J(Theta) Cost')
# ax.set_xlabel('Iterations')
# #_ = ax.plot(range(no_iterations), cost_history, 'b.')
#
# # zoom 200
# _ = ax.plot(range(200), cost_history[:200], 'b.')
#
# plt.show()




#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
#this is just to demonstrate gradient descent

from numpy import *
from matplotlib import pylab as plt

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))




if __name__ == '__main__':
    run()

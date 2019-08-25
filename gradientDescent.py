import numpy as np
import matplotlib.pylab as plt
plt.style.use(['ggplot'])

# slope equation y = b + mx

b = 4
m = 3

X = 2 * np.random.rand(100, 1)
y = b+m*X + np.random.randn(100, 1)

# plot our data with reltion to x and y
plt.plot(X, y, 'b.')
plt.xlabel("$X$", fontsize=18)
plt.ylabel("Y", rotation=0, fontsize=18)
_ = plt.axis([0, 2, 0, 15])



# Calculate cost function or MSE
def cal_cost(theta, X, y):

	m = len(y)

	predictions = X.dot(theta)
	cost = (1/2*m) * np.sum(np.square(predictions - y))

	return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
	m = len(y)
	cost_history = np.zeros(iterations)
	theta_history   = np.zeros((iterations, 2))

	for it in range(iterations):
		prediction = np.dot(X, theta)
		theta = theta - learning_rate * (1/m) *(X.T.dot((prediction - y)))
		theta_history[it, :] = theta.T
		cost_history[it] = cal_cost(theta, X, y)
	return theta, cost_history, theta_history


#plt.show()
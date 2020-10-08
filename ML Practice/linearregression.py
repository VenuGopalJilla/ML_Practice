import pandas as pd
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour 

data = pd.read_excel("toyota corolla.xlsx")
df = pd.DataFrame(data)
# print(df.columns)


scatter(df['Age_08_04'][:10], df['Price'][:10], marker = 'X', c = 'r')
title("Price of Cars")
xlabel("Age of the Car")
ylabel("Price of the Car based on age")
show()


X = df['Age_08_04']
Y = df['Price']


#number of training examples
m = Y.size
# print(m)


#adding 1 column of 1's to X (intercept values)
it = ones(shape = (m, 2))
it[:, 1] = X

# print(it)

theta = zeros(shape=(2, 1))
iterations = 1000
alpha = 0.001



def Compute_Cost(X, Y, theta):
    m = Y.size

    predictions = X.dot(theta).flatten()

    sq_errors = (predictions - Y)**2

    J = (1 / (2 * m)) * (sq_errors.sum())

    return J


def gradient_descent_univariate(X, Y, theta, alpha, num_iters):
    m = Y.size

    J_history = zeros(shape = (num_iters, 1))

    for i in range(num_iters):
        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - Y) * X[:, 0]
        errors_x2 = (predictions - Y) * X[:, 1]

        theta[0][0] = theta[0][0] - (alpha * (1.0 / m) * (errors_x1.sum()))
        theta[1][0] = theta[1][0] - (alpha * (1.0 / m) * (errors_x2.sum()))

        print(theta[0][0], theta[1][0])

        J_history[i, 0] = Compute_Cost(X, Y, theta)

    return theta, J_history


theta, J_hstory = gradient_descent_univariate(it, Y, theta, alpha, iterations)
print(theta, " --- >  Theta Values")



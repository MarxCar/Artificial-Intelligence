import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math


dataset = sio.loadmat("ex3data1.mat")
X = np.c_[np.ones(dataset['X'].shape[0]),dataset['X']]
y = np.c_[dataset['y']]

plt.imshow(X[np.random.choice(X.shape[0],20), 1:].reshape(-1,20).T)
plt.axis('off')
#plt.show()

def sigmoid(z):
	return (1/(1+np.power(math.e,-z)))


def cost_function(x, y, theta, regular):
    m = len(y)
    hypothesis = sigmoid(x.dot(theta))
    J = (1.0 / m) * (np.log(hypothesis).T.dot(y) + (np.log(1 - hypothesis)).T.dot(1 - y))
    reg = (regular / (2 * m)) * np.sum(np.square(theta[1:]))
    regularized = (J+reg)
    if np.isnan(regularized[0]):
        return np.inf
    return regularized[0]


def gradient_descent(theta,x,y, regular):
    hypo = sigmoid(x.dot(theta))
    m = len(y)
    return (((1.0/m) * (x.T.dot(hypo - y))) + (regular/m)*theta).flatten()



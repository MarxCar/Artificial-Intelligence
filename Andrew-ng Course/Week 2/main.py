import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = np.loadtxt("ex1data1.txt", delimiter=",")




plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")


x = np.c_[np.ones(df.shape[0]),df[:,0]]
y = np.c_[df[:,1]]

theta = np.array([[0.],[0.]])
iterations = 1500
alpha = 0.01
m = y.size

plt.scatter(x[:,1],y)

def hypothesis(theta,X):
	return np.dot(X, theta)

def mean_squared_error(t,c,b):
	m = len(c)
	error = hypothesis(t,c) - b

	return float((1.)/(2*m)*np.dot(error.T, error))



def gradient_descent_runner(w, xs, ys, a, frequency):
	m = len(xs)
	temp = []
	for f in range(frequency):

		for t in range(len(w)):
	
			temp.append(sum([xs[i,t]*(hypothesis(w,xs[i]) -ys[i]) for i in range(m)]))
		
		for k in range(len(temp)):
			w[k,0] = w[k,0] - a*(temp[k]/m)
	
		temp = []
		
	return w
def line(theta, xvalues):
	return theta[0] + theta[1]*xvalues


print("Error before gradient_descent: %f" % mean_squared_error(theta,x,y))

theta = gradient_descent_runner(theta, x,y,alpha,iterations)

print("Final Weights: " + str(theta))
print("Final Error: %f" %(mean_squared_error(theta,x,y)))



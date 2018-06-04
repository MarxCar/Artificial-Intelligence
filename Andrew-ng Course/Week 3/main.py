import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
data = np.loadtxt("ex2data1.txt", delimiter=",")


X = np.c_[np.ones(data.shape[0]),data[:,:2]]
y = np.c_[data[:,-1]]
iterations = 1500
alpha = 0.01
m = len(X)

plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")


for i in range(m):
	yvalue = data[i,-1]
	if yvalue:
		color, marker= "black", "+"
	else:
		color,marker = "yellow", "o"
	plt.scatter(data[i,0], data[i,1], c=color, marker=marker)


#plt.show()


def sigmoid(z):
	return (1/(1+np.power(math.e,-z)))

def hypothesis(xval,t):
	return sigmoid(np.dot(xval,t))

def costFunction(x,y, theta):
	c = hypothesis(x,theta)
	return (np.dot(-y.T,np.log(c)) -np.dot(1-y.T, np.log(1-c)))/len(x)

def costFunctionRunner(t):
	return costFunction(X,y,t)

def gradientDescent(x,y,a,generations):
	t = np.zeros((x.shape[1],1))
	m = len(x)
	for k in range(generations):
		temp =  np.dot(x.T, hypothesis(x,t)-y)
		t = t - (a/m) * temp
	return t




theta =  np.array([[0],[0],[0]])
print("Cost of Model Before Hand: " + str(costFunction(X,y,theta)[0]))


#minimize cost function using SLSQP method instead of gradient descent
res = minimize(costFunctionRunner, theta, method="SLSQP")
new_thetas = res.x
print(new_thetas)
print("Cost of Model After SLSQP: " + str(res.fun))

print(gradientDescent(X,y,0.001, 10000))
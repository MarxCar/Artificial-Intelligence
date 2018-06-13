import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from sklearn import linear_model
data = np.loadtxt("ex2data1.txt", delimiter=",")


X = np.c_[np.ones(data.shape[0]),data[:,:2]]

y = np.c_[data[:,2]]
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




def sigmoid(z):
	return (1/(1+np.power(math.e,-z)))

def hypothesis(xval,t):
	return sigmoid(np.dot(xval,t))

def costFunction(theta,x,y):
	c = hypothesis(x,theta)
	J = (np.dot(-y.T,np.log(c)) -np.dot(1-y.T, np.log(1-c)))/len(x)

	if np.isnan(J[0]):
		return(np.inf)
	return(J[0])

def gradientDescent(t,x,y):
	t = t.reshape(-1,1)
	m = len(x)
	temp =  np.dot(x.T, hypothesis(x,t)-y)
	gradient = (1.0/m) * temp

	return(gradient.flatten())


theta =  np.array([[0],[0],[0]])
print("Cost of Model Before Hand: " + str(costFunction(theta,X,y)[0]))


#minimize cost function using SLSQP method instead of gradient descent
res = minimize(costFunction, theta, args=(X,y), method="BFGS")
new_thetas = res.x


print(new_thetas)
print("Cost of Model After SLSQP: " + str(res.fun))

#gradient descent
res2 = minimize(costFunction, theta, args=(X,y), method=None, jac=gradientDescent, options={'maxiter':400})

print("Cost of Model After Gradient Descent:  " + str(res2.fun))






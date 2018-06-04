import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
data = np.loadtxt("ex2data1.txt", delimiter=",")


X = np.c_[np.ones(data.shape[0]),data[:,:2]]
y = np.c_[data[:,-1]]
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



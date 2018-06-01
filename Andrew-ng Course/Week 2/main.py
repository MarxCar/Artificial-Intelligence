import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("ex1data1.txt", delimiter=",")
df.columns = ["population", "price"]
m = len(df.index)


plt.scatter(df.population,df.price)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

x = np.array([1]*m)
x = np.c_[x,df.as_matrix()]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01


print(x)



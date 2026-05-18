#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
x= np.random.uniform(0, 100, size=1000)
noise = np.random.normal(0, 10, size=1000)
y = 4 * x + 7 + noise
m=x.size
alpha=0.0001
w=0
b=0
cost=[]
cost_function=(1/(2*m))*(np.sum((w*x+b-y)**2))
for i in range(0,100000):
    predict=(w*x+b-y)
    gradient=(alpha/m)*(np.sum(predict*x))
    w=w-gradient
    b=b-(alpha/m)*(np.sum(predict))
    cost_function=(1/(2*m))*(np.sum((w*x+b-y)**2))
    cost.append(cost_function)


cost_f=np.array(cost)
plt.plot(range(0, 100000), cost_f, color='blue', label='Cost')
plt.yscale('log')
plt.title("Iteration vs Cost function")
plt.xlabel("Number of Iteration")
plt.ylabel("Cost Function")
plt.grid(True)
plt.legend()
plt.show()

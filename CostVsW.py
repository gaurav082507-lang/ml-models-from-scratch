#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,4,5,6])
m=x.size;
w=np.linspace(-4,4,300)
cost_list=[]

for i in w:
    cost=(1/2*m)*(np.sum((i*x-y)**2))
    cost_list.append(cost)

cost_function=np.array(cost_list)
plt.figure(figsize=(10,10))
plt.title("Cost Function Vs W Parameter")
plt.plot(w,cost_function)
plt.xlabel("Parameter W")
plt.ylabel("Cost Function ")
plt.grid(True)
plt.legend()
plt.show()

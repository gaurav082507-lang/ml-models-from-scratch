#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11])
y = np.array([0,0,0,0,1,0,1,0,1,1,0,1,1,1,1,1, 1, 1, 1, 1])
w=np.linspace(-3,8,500)
m=x.size
x1_norm=(x-np.mean(x))/np.std(x)
cost_function=[]
for i in w:
    f1=i*x1_norm
    f=1/(1+np.exp(-f1))
    cost=(-1/m)*(np.sum(y*np.log(f)+(1-y)*np.log(1-f)))
    cost_function.append(cost)

cost_f=np.array(cost_function)
plt.figure(figsize=(10,10))
plt.title("Cost Function vs W parameter")
plt.xlabel("W parameter")
plt.ylabel("Cost Function")
plt.plot(w,cost_f)
plt.grid(True)
plt.show()




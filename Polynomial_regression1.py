#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
x = np.array([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90], dtype=float)
y = np.array([110,112,115,118,122,125,130,135,140,145,150,155,160,165,170], dtype=float)
m=x.size
x1=x
x2=x**2
# feature Scaling 
x1_norm=(x1-np.mean(x1))/np.std(x1)
x2_norm=(x2-np.mean(x2))/np.std(x2)
y_norm=(y-np.mean(y))/np.std(y)

w1=0.0
w2=0.0
b=0.0
alpha=0.01
for i in range(0,10000):
    pre=w1*x1_norm+w2*x2_norm+b-y_norm
    tempw1=w1-(alpha/m)*(np.sum(pre*x1_norm))
    tempw2=w2-(alpha/m)*(np.sum(pre*x2_norm))
    tempb=b-(alpha/m)*(np.sum(pre))
    w1=tempw1
    w2=tempw2
    b=tempb

x_plot = np.linspace(20, 90, 300)
y1=w1*x1_norm+w2*x2_norm+b
plt.title("X vs Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x1_norm,y_norm,marker='x')
plt.plot(x1_norm, y1, label=f'{w2:.4f}*x²+{w1:.4f}*x+{b:.4f}')
plt.grid(True)
plt.legend()
plt.show()




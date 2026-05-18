#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3, 3, 300)
y=1/(1+np.exp(-x))
plt.figure(figsize=(8,8))
plt.plot(x,y)
plt.title("Sigmoid Function")
plt.xlabel("X")
plt.ylabel("Sigmoid Function(Y)")
plt.grid(True)
plt.show()

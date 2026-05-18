#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0.001, 0.999, 300)
y=np.log(x)
y1=np.log(1-x)
y1=-y1
y=-y

plt.figure(figsize=(8,8))
plt.title("Loss Function Vs Z")
plt.plot(x, y,color='red',label="Loss Function when y=1")
plt.plot(x, y1,color='blue',label="Loss Function when y=0")
plt.xlabel("Z")
plt.ylabel("Loss Function")
plt.legend()
plt.grid(True)
plt.show()
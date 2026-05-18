import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
x = np.linspace(-3, 3, 200)
noise = np.random.normal(0, 15, 200)
# y = 2x⁵ - 3x⁴ - 10x³ + 15x² + 5x - 8 + noise
y = 2*x**5 - 3*x**4 - 10*x**3 + 15*x**2 + 5*x - 8 + noise
m=x.size
x2=x**2
x3=x**3
x4=x**4
x5=x**5
x1_norm=(x-np.mean(x))/np.std(x)
x2_norm=(x2-np.mean(x2))/np.std(x2)
x3_norm=(x3-np.mean(x3))/np.std(x3)
x4_norm=(x4-np.mean(x4))/np.std(x4)
x5_norm=(x5-np.mean(x5))/np.std(x5)
y_norm=(y-np.mean(y))/np.std(y)
w1=0
w2=0
w3=0
w4=0
w5=0
b=0
alpha=0.01
for i in range(0,100000):
    f=w1*x1_norm+w2*x2_norm+w3*x3_norm+w4*x4_norm+w5*x5_norm+b
    error=f-y_norm
    w1=w1-(alpha/m)*(np.sum(error*x1_norm))
    w2=w2-(alpha/m)*(np.sum(error*x2_norm))
    w3=w3-(alpha/m)*(np.sum(error*x3_norm))
    w4=w4-(alpha/m)*(np.sum(error*x4_norm))
    w5=w5-(alpha/m)*(np.sum(error*x5_norm))
    b=b-(alpha/m)*(np.sum(error))

y_f=w5*x5_norm+w4*x4_norm+w3*x3_norm+w2*x2_norm+w1*x1_norm+b
plt.figure(figsize=(10, 6))
# x_plot=np.linspace(np.min(x1_norm),np.max(x1_norm),500)
plt.title("Polynomial Regression")
plt.scatter(x1_norm,y_norm,marker='x',s=5)
plt.plot(x1_norm,y_f,color='blue',label=f'{w5:.2f}*x^5+{w4:.2f}*x^4+{w3:.2f}*x^3+{w2:.2f}*x^2+{w1:.2f}*x+{b:.2f}')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

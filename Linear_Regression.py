import numpy as np
import matplotlib.pyplot as plt
#-----------------Data Set--------------------------
np.random.seed(17)
x = np.random.uniform(0, 50, size=1000)
noise = np.random.normal(0, 5, size=1000)
y = (lambda x, n: 6.3 * x + 42.7 + n)(x, noise)
#---------Parameters--------------------------
m=x.size
alpha=0.00003
w=0
b=0
#-------------Gradient Descent--------------------
for i in range(0,2000000):
    predict=(w*x+b-y)
    gradient=(alpha/m)*(np.sum(predict*x))
    w=w-gradient
    b=b-(alpha/m)*(np.sum(predict))

    if i % 500000 == 0:
        cost = (1/(2*m)) * np.sum(predict**2)
        print(f"Iter {i:>8} | cost={cost:.4f} | w={w:.4f} | b={b:.4f}")

# ------Plot--------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='red', alpha=0.5, label='Data points')
plt.axline((0, b), slope=w, color='blue', linewidth=2,
           label=f'Predicted: y = {w:.2f}x + {b:.2f}')
plt.axline((0, 42.7), slope=6.3, color='green', linewidth=2,
           linestyle='--', label='True: y = 6.3x + 42.7')
plt.title("Linear Regression From Scratch")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

#Prediction Column

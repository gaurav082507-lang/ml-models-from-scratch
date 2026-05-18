#! C:\Users\gaura\OneDrive\Desktop\Python\myenv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
# Feature 1: Size of House (sq ft)
x1 = np.random.uniform(500, 5000, size=1000)
# Feature 2: Number of Bedrooms
x2 = np.random.randint(1, 6, size=1000)
# Combine features into matrix X (1000 x 2)
X = np.column_stack((x1, x2))
noise = np.random.normal(0, 10, size=1000)
# Output: House Price in $1000
y = 3 * x1 + 20 * x2 + 50 + noise

# Feature Scaling 

x1_norm = (x1 - np.mean(x1)) / np.std(x1)
x2_norm = (x2 - np.mean(x2)) / np.std(x2)
alpha=0.01
w1=0
w2=0
b=0
for i in range(0,10000):
    temp1=w1-(alpha/1000)*(np.sum((w1*x1_norm+w2*x2_norm+b-y)*x1_norm))
    temp2=w2-(alpha/1000)*(np.sum((w1*x1_norm+w2*x2_norm+b-y)*x2_norm))
    temp3=b-(alpha/1000)*(np.sum(w1*x1_norm+w2*x2_norm+b-y))
    w1=temp1
    w2=temp2
    b=temp3


print(f"w1: {w1:.4f}, w2: {w2:.4f}, b: {b:.4f}")

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, s=2, color='red', label='Actual Data')

# ✅ Normalize the grid before predicting
x1_range = np.linspace(500, 5000, 50)
x2_range = np.linspace(1, 6, 50)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x1_grid_norm = (x1_grid - np.mean(x1)) / np.std(x1)
x2_grid_norm = (x2_grid - np.mean(x2)) / np.std(x2)
y_grid = w1 * x1_grid_norm + w2 * x2_grid_norm + b

ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='blue', label='Regression Plane')

# ✅ Convert weights back to original scale
w1_real = w1 / np.std(x1)
w2_real = w2 / np.std(x2)
b_real  = b - w1 * np.mean(x1) / np.std(x1) - w2 * np.mean(x2) / np.std(x2)
ax.set_xlabel("House Size (sq ft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price ($1000)")
ax.set_title(f"3D Linear Regression\ny = {w1_real:.4f}*x1 + {w2_real:.4f}*x2 + {b_real:.4f}")
plt.legend()
plt.show()
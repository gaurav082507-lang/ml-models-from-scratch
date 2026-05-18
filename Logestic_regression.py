import numpy as np
import matplotlib.pyplot as plt
#--- Data Set-----------------------------------------------
x = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1100], dtype=float)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 1, 1, 1, 1, 1, 1, 1,  1], dtype=float)
x1_norm=(x-np.mean(x))/np.std(x)
#------- Hyper Parameters-----------------
alpha=0.01
w=0
b=0
m=x.size
#---------------Gradient Descent---------------------------
for i in range(0,500000):
    f=1/(1+np.exp(-(w*x1_norm+b)))
    error    = f - y
    w = w - (alpha/m) * np.sum(error * x1_norm)
    b = b - (alpha/m) * np.sum(error)

    if(i%10000==0):
        cost=(1/(2*m))*(np.sum((f-y)**2))
        print(f"Iteration : {i} |Cost : {cost:2f} | w :{w:2f}| b :{b:2f}")
#------------------Plot--------------------------------------------------------------
print(f"Emails Above length {round((-b/w)*np.std(x)+np.mean(x))} are spam")
x_plot=np.linspace(np.min(x1_norm),np.max(x1_norm),300)
sigmoid=1/(1+np.exp(-(w*x_plot+b)))
plt.figure(figsize=(12,12))
plt.xlabel("Email Length Normalized")
plt.ylabel("Spam OR not Spam ")
plt.scatter(x1_norm[y==1],y[y==1],marker='x',color='red')
plt.axvline(x=-b/w, color='red', linestyle='--',label=f'x=-{(b/w):2f}')
plt.scatter(x1_norm[y==0],y[y==0],marker='o',color='blue')
plt.plot(x_plot,sigmoid)
plt.grid(True)
plt.legend()
plt.show()

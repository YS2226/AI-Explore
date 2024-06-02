import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)


plt.scatter(x, y, color='blue', label='Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


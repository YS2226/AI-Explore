import numpy as np  
from dezero import Variable

x_np = np.array(5.0)
x = Variable(x_np)


y = 3*x**2
print(y)

y.backward()
print(x.grad)
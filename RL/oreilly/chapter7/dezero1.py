import numpy as np
from dezero import Variable
import dezero.functions as F

a = np.array([1,2,3])
b = np.array([4,5,6])
a,b = Variable(a),Variable(b)

c=F.matmul(a,b)
print(c)

a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
c=F.matmul(a,b)
print(c)


#Rosenbrok

def rosenbrock(x,y):
    z = 100*(y-x**2)**2 + (x-1)**2
    return z
x0=Variable(np.array(0.0))
x1=Variable(np.array(2.0))
y = rosenbrock(x0,x1)
y.backward()
print(x0.grad,x1.grad)


# 勾配降下法

x0=Variable(np.array(0.0))
x1=Variable(np.array(2.0))

lr=0.001
iters=10000

for i in range(iters):
    y=rosenbrock(x0,x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data -=lr*x0.grad.data
    x1.data -=lr*x1.grad.data

print(x0,x1)

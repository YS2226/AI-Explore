import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)
x,y=Variable(x),Variable(y)

W=Variable(np.zeros((1,1)))
b=Variable(np.zeros(1))

def predict(x):
    y=F.matmul(x,W)+b
    return y

def mean_squared_error(x0,x1):
    diff=x0-x1
    return F.sum(diff**2)/len(diff)

lr=0.1
iters=100

for i in range(iters):
    y_pred=predict(x)
    loss=mean_squared_error(y,y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data-=lr*W.grad.data
    b.data-=lr*b.grad.data

    if i %10==0:
        print(loss.data)

print("====================")
print("W: %s"%W.data)
print("b: %s"%b.data)

plt.scatter(x.data, y.data, color='blue', label='Data Points')
x_line = np.linspace(0, 1, 100).reshape(100, 1)
y_line = predict(Variable(x_line)).data
plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
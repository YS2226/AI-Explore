import numpy as np
import dezero.layers as L
from dezero import Model
import dezero.functions as F
from dezero import optimizers



class TwoLayerNet(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self,x):
        y = F.relu(self.l1(x))
        y=self.l2(y)
        return y 
    
np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+ np.random.rand(100,1)
lr=0.2
iters=10000

model = TwoLayerNet(10,1)

optimizer= optimizers.SGD(lr)

for i in range(10,1):
    y_pred=model.forward(x)
    loss=F.mean_squared_error(y,y_pred)
    model.cleargrads()
    loss.backward()
    
    optimizer.update()
    if i%1000==0:
        print(loss)



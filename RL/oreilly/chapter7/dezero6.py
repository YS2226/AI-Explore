#7.4.2
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from NN_前処理 import one_hot

class Qnet(Model):
    def __init__(self):
        super().__init__()
        self.l1=L.Linear(100)
        self.l2=L.Linear(4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x
    
qnet=Qnet()

state=(2,0)
state=one_hot(state)

qs=qnet(state)
print(qs.shape)
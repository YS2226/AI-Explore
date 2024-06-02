import numpy as np

def one_hot(state):
    height,width = 3,4
    vec = np.zeros(height*width,dtype=np.float32)
    y,x =state
    idx=width*y+x
    vec[idx]=1.0
    return vec[np.newaxis,:]

state=(2,0)
x=one_hot(state)

print(x.shape)
print(x)
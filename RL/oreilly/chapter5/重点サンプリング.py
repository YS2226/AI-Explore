import numpy as np

x=np.array([1,2,3])
pi=np.array([0.1,0.1,0.8])

#期待値
e = np.sum(x*pi)
print("Expected val: ",e)

#モンテカルロ
#１００個サンプリングして平均をとる
n=100
samples=[]
for i in range(n):
    #piの確率でxをサンプリング
    #p=piとしないと  エラーが出る
    #                 s=np.random.choice(x,pi)     
    #                 ^^^^^^^^^^^^^^^^^^^^^
    #TypeError: 'numpy.float64' object cannot be interpreted as an integer  
    s=np.random.choice(x,p=pi)
    samples.append(s)
mean = np.mean(samples)
var=np.var(samples)
print("MC: {:.2f} (var: {:.2f})".format(mean,var))

b=np.array([1/3,1/3,1/3])
n=100
samples=[]

#重点サンプリングを試す＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿

for _ in range(n):
    idx=np.arange(len(b)) #0,1,2
    i = np.random.choice(idx,p=b)

    s = x[i]

    # p ( pi(x) / b(x)  ) 
    rho = pi[i] / b[i]
    samples.append(rho*s)

mean=np.mean(samples)
var=np.var(samples)
print("IS: {:.2f} (var: {:.2f})".format(mean,var))



#分散を小さくするには

b=np.array([0.2,0.2,0.6])
n=100
samples=[]

for _ in range(n):
    idx=np.arange(len(b)) #0,1,2
    i = np.random.choice(idx,p=b)

    s = x[i]

    # p ( pi(x) / b(x)  ) 
    rho = pi[i] / b[i]
    samples.append(rho*s)

mean=np.mean(samples)
var=np.var(samples)
print("IS2: {:.2f} (var2: {:.2f})".format(mean,var))

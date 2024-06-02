from dezero6 import Qnet
from dezero import optimizers
import numpy as np
import dezero.functions as F
from NN_前処理 import one_hot
import sys
sys.path.append("C:/Users/YS/Desktop/RL/scratch/chapter4")
from gridworld import GridWorld
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self):
        self.gamma=0.9
        self.lr=0.01
        self.epsilon=0.1
        self.action_size=4

        self.qnet=Qnet()
        self.optimizer=optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)
    
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs=self.qnet(state)
            return qs.data.argmax()
    
    def update(self,state,action,reward,next_state,done):
        if done:
            next_q=np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()
            """
            Q-learningでは次の状態の最大Q値 (next_q) をターゲットQ値 (target) として使用しますが、
            その最大Q値自体の計算において勾配を計算したくありません。これを実現するために unchain() を使っています。

            chatGPTより
            """
        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target,q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data
    
env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss,cnt = 0,0
    done=False

    while not done:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)

        next_state = one_hot(next_state)

        loss=agent.update(state,action,reward,next_state,done)
        total_loss += loss
        cnt +=1
        state = next_state
    
    average_loss = total_loss /cnt
    loss_history.append(average_loss)
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# visualize
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)

#render はディクショナリを受け取る
env.render_q(Q)

    
    
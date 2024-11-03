import copy
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from replay_buffer import ReplayBuffer
import numpy as np
import gym

class QNet(Model):
    def __init__(self,action_size):
        super().__init__()# ensuring that the initialization code in the Model class is executed when you create an instance of QNet
                        # This allows QNet to access action_size information which is in DQNAgent class.
        self.l1=L.Linear(128) #creating a linear transformation layer in the neural network with 128 output units. 
        self.l2=L.Linear(128)
        self.l3=L.Linear(action_size)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=self.l3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.gamma=0.9
        self.lr=0.0005
        self.epsilon = 0.1
        self.buffer_size=10000
        self.batch_size = 32
        self.action_size=2 # 最終的なアクションは右に行くか左に行くかの２択のため。

        self.replay_buffer = ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target=QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)# only creating instance of Adam optimizer, not connected to network itself. 
        self.optimizer.setup(self.qnet)#linking the optimizer to the neural network model. 

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
        """
        copy.copy vs copy.deepcopy

        浅いコピーは同じものを違う変数に設定して中身を変えるとどちらの変数も変更される
        深いコピーは一回コピーしてしまえばお互いのデータが共有されることはない

        以下に詳しい説明を残す
        https://chatgpt.com/share/9603692c-4089-469a-94a1-f3ae7b9e397b

        """
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        else:
            state=state[np.newaxis,:] #バッチの次元を追加
            qs=self.qnet(state)
            return qs.data.argmax()
    
    def update(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)# 経験の蓄積
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state,action,reward,next_state,done = self.replay_buffer.get_batch() #ミニバッチサイズ以上のデータがたまったらミニバッチを作成
        qs = self.qnet(state) # 状態は（３２、４）のnp.ndarray。　３２個分のデータをまとめてネットワークにぶち込む 出力はサイズ２の形状
        q = qs[np.arange(self.batch_size), action]


        next_qs=self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)# ターゲットネットワークを使用してSt+1を求める　そしてその最大値を求める
        next_q.unchain()
        target=reward + (1-done) * self.gamma*next_q

        loss = F.mean_squared_error(q,target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


# カートポール　vs DQN

count = 100
episodes = 300
sync_interval =20
env = gym.make("CartPole-v0")
agent = DQNAgent()

reward_history = []

for iter in range(count):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward,done,info = env.step(action)

            agent.update(state,action,reward,next_state,done)
            state =next_state
            total_reward += reward
        if episode % sync_interval == 0:
            agent.sync_qnet()
        
        reward_history.append(total_reward/(iter+1))
        

import matplotlib.pyplot as plt

# Assuming reward_history is populated as in the previous code
episodes = 300

plt.plot(range(episodes), reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode vs Total Reward')
plt.show()

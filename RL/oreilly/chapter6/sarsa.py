from collections import defaultdict,deque
import numpy as np
import sys
sys.path.append("C:/Users/YS/Desktop/RL/scratch")
from utils import greedy_probs
sys.path.append("C:/Users/YS/Desktop/RL/scratch/chapter4")
from gridworld import GridWorld


# SARSA: ヨーするにTD法でやったものを価値関数ではなくてQ関数で実行してそれをεgreedyによって方策評価と改善の繰り返しを行っていくプロセス。
#SARTA__________________________________________________________________________________________________________________________________

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda:random_actions)
        self.Q = defaultdict(lambda:0)
        self.memory = deque(maxlen=2)
        #deque 
        # リストと同様に使うことができる。最大要素数を超えるdeque使うことで直近の経験データ２つ分だけを保持。

    def get_action(self,state):
        action_probs=self.pi[state] #piから選ぶ
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    def reset(self):
        self.memory.clear()
    def update(self,state,action,reward,done):
        self.memory.append((state,action,reward,done))
        if len(self.memory)<2:
            return
        
        state,action,reward,done = self.memory[0]
        next_state,next_action,_,_ = self.memory[1]
        #次のQ関数
        next_q = 0 if done else self.Q[next_state,next_action]
        #TD法の更新
        target=reward + self.gamma*next_q
        self.Q[state,action] +=(target - self.Q[state,action]) * self.alpha

        #方策の改善
        self.pi[state] = greedy_probs(self.Q,state,self.epsilon)

env=GridWorld()
agent=SarsaAgent()
episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward,done = env.step(action)
        agent.update(state,action,reward,done)
        if done:
            #ゴールに到達したときにも呼ぶ
            agent.update(next_state,None,None,None)
            break
        state = next_state
env.render_q(agent.Q)


#SARSA test end ________________________________________________________________________________________________________________________




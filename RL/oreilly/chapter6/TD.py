from collections import defaultdict
import numpy as np
import sys
sys.path.append("C:/Users/YS/Desktop/RL/scratch/chapter4")
from gridworld import GridWorld

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        #方策の初期化
        self.pi = defaultdict(lambda:random_actions)
        self.V = defaultdict(lambda: 0)
    #状態を受け取り方策に基づいて行動（この場合はランダム)を返す
    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    def eval(self,state,reward,next_state,done):
        next_V = 0 if done else self.V[next_state]#ゴールの価値関数を０に
        target = reward + self.gamma*next_V
        #価値関数の更新：モンテカルロでは収益が確定してから方策が更新されてたけどTD法は動くたびに更新されるため、何度も更新用の関数を呼び出す。
        self.V[state] +=(target - self.V[state])*self.alpha

env = GridWorld()
agent = TdAgent()
episodes =1000
for episode in range(episodes):
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.eval(state,reward,next_state,done) #毎回呼ぶ
        if done:
            break
        state = next_state
env.render_v(agent.V)

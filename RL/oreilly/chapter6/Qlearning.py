from collections import defaultdict
import numpy as np
import sys
sys.path.append("C:/Users/YS/Desktop/RL/scratch")
from utils import greedy_probs
sys.path.append("C:/Users/YS/Desktop/RL/scratch/chapter4")
from gridworld import GridWorld

class QlearningAgent:
    def __init__(self):
        self.gamma=0.9
        self.alpha=0.8
        self.epsilon=0.1
        self.action_size=4

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda:random_actions)#ターゲット方策
        self.b = defaultdict(lambda:random_actions) #行動方策
        self.Q = defaultdict(lambda:0) 
    
    def get_action(self,state):
        action_probs=self.pi[state] #piから選ぶ
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    
    def update(self,state,action,reward,next_state,done):
        if done:
            next_q_max=0
        else:
            #行動価値関数を計算
            next_qs = [self.Q[next_state,a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        target=reward+self.gamma*next_q_max

        #行動価値関数の更新
        self.Q[state,action]+= (target - self.Q[state,action]) *self.alpha

        #方策の更新
        self.pi[state]=greedy_probs(self.Q,state,epsilon=0)
        self.b[state]=greedy_probs(self.Q,state,self.epsilon)

#動かしてみる
env=GridWorld()
agent=QlearningAgent()
import gym
from IPython.display import clear_output
env = gym.make('Acrobot-v1')
num_episodes = 500

rewards = []
# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.update(state,action,reward,next_state,done)
        if done:
            #SARSAと違いゴールに到達したらそれ以上のアクションはいらないので特にない
            break
        state = next_state

        # 可視化のためのエピソード
        if episode % 100 == 0:
            clear_output(wait=True)
            env.render()
    rewards.append(total_reward)
env.close()

        
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()


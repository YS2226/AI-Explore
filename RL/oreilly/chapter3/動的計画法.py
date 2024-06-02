V = {'L1': 0.0,'L2':0.0}

"""
for i in range(100):
    new_V['L1'] = 0.5*(-1+0.9*V['L1']) + 0.5*(1+0.9*V['L2'])
    new_V['L2'] = 0.5*(0+0.9*V['L1']) + 0.5*(-1+0.9*V['L2'])
    V=new_V.copy()
    print(V)
"""
"""
cnt=0
while True:
    new_V['L1'] = 0.5*(-1+0.9*V['L1']) + 0.5*(1+0.9*V['L2'])
    new_V['L2'] = 0.5*(0+0.9*V['L1']) + 0.5*(-1+0.9*V['L2'])
    
    #閾値を設定して価値関数を更新の伸びが閾値より小さくなったらストップ
    delta = abs(new_V['L1']- V['L1'])
    delta = max(delta, abs(new_V['L2']- V['L2']))
    V=new_V.copy()
    cnt+=1
    if delta <0.0001:
        print(V)
        print(cnt)
        break
"""
'''
cnt=0
while True:
    t = 0.5*(-1+0.9*V['L1']) + 0.5*(1+0.9*V['L2'])
    delta =abs(t - V['L1'])
    V['L1']=t
    t = 0.5*(0+0.9*V['L1']) + 0.5*(-1+0.9*V['L2'])
    delta =max(delta,abs(t - V['L2']))
    V['L2']=t
    cnt+=1
    if delta <0.0001:
        print(V)
        print(cnt)
        break
'''
'''
#actionとstateの確認
for action in env.actions():
    print(action)

print('===')
for state in env.states():
    print(state)   
'''

#環境設定____________________________________________________________________________________________________________________
#ここからは３ｘ４のGridWorldクラスの実装に移る　p.97 5/2/24
import numpy as np 
import gridworld_render as render_helper

class GridWorld:
    def __init__(self):
        self.action_space=[0,1,2,3]
        self.action_meaning = {
            0: "UP",
            1:"DOWN",
            2:"LEFT",
            3:"RIGHT",
        }
        self.reward_map = np.array(
            [[0,0,0,1.0],
             [0,None,0,-1.0],
             [0,0,0,0]
             ]
        )
        #ゴールが(3,0)
        self.goal_state=(0,3)
        self.wall_state=(1,1)
        #スタートは(2,0)
        self.start_state=(2,0)
        self.agent_state=self.start_state

    @property 
    def height(self):
        return len(self.reward_map)
    @property
    def width(self):
        return len(self.reward_map[0])
    @property
    def shape(self):
        return self.reward_map.shape
    def actions(self):
        return self.action_space # 0,1,2,3
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield h,w
    #環境の状態遷移
    def next_state(self,state,action):
        #移動先の場所の計算
        action_move_map = [(-1,0),(1,0),(0,-1),(0,1)]
        move=action_move_map[action]
        next_state=(state[0]+move[0],state[1]+move[1])
        ny,nx=next_state
        #移動先が壁化枠外かの確認
        if nx<0 or nx>=self.width or ny>0 or ny>=self.height:
            next_state=state
        elif next_state==self.wall_state:
            next_state=state
        return next_state
    def reward(self,state,action,next_state):
        reward_value = self.reward_map[next_state]
        if reward_value is None:
            return 0  # or any default value that makes sense in your context
        return reward_value
    #以下コピペ
    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)

#環境設定終わり_________________________________________________________________________________________________
'''
env = GridWorld()
V={}
for state in env.states():
    V[state]=np.random.randn()
env.render_v(V)
'''
#____________________________________________________________________________________________
#####方策反復法
from collections import defaultdict
env = GridWorld()
'''
#test defaultdict/ state is tuple. So it will return error for normal dictionary. In this defaultdict, it will return 0 
state = (1,2)
print(V[state]) #0
'''

pi=defaultdict(lambda: {0:0.25,1:0.25,2:0.25,3:0.25})
state=(0,1)

def eval_onestep(pi,V,env,gamma=0.9):

    for state in env.states():# 各状態へのアクセス
       
        if state==env.goal_state: #ゴールの価値関数は０
            V[state]=0
            continue
        action_probs=pi[state] 
        new_V=0
        for action,action_prob in action_probs.items():
            next_state=env.next_state(state,action)
            r=env.reward(state,action,next_state)
            new_V += action_prob*(r + gamma*V[next_state])
        V[state]=new_V
    return V
def policy_eval(pi,V,env,gamma,threshold=0.001):
    
    while True:
        old_V=V.copy()
        V=eval_onestep(pi,V,env,gamma)
        #更新された量の最大値を求める
        delta=0
        for state in V.keys():
            t=abs(V[state]-old_V[state])
            if delta<t:
                delta=t
        #閾値との比較
        if delta<threshold:
            break
    return V

#####方策反復法終わり____________________________________________________________________________________________________

'''
#価値関数の可視化
env=GridWorld()
gamma=0.9
V=defaultdict(lambda:0)
V=policy_eval(pi,V,env,gamma)
env.render_v(V,pi)
'''
#状態価値関数のため、それぞれのマスの”価値”を測っている。ゴールについて初めて報酬がもらえるため、ゴールの一個手前のマスの価値関数がわかって初めて
#価値関数は更新される。そこから一個ずつ逆算されることですべてのマスの価値関数がわかるわけだ。
####またエージェントは動いていない。動くのは行動価値関数のほう。

#方策の改善

def argmax(d):
    max_value = max(d.values())
    max_key=0
    for key, value in d.items():
        if value == max_value:
            max_key=key
    return max_key
#greedy化のための関数
def greedy_policy(V,env,gamma):
    pi={}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_state=env.next_state(state,action)
            #報酬関数
            r=env.reward(state,action,next_state)
            #報酬の総和
            value= r+ gamma *V[next_state]
            #それぞれの行動に対して報酬をあてるディクショナリー
            action_values[action]=value
        #方策のgreedy化
        max_action = argmax(action_values)
        action_probs={0:0,1:0,2:0,3:0}
        action_probs[max_action]=1.0
        pi[state]=action_probs
    return pi
#評価と改善の繰り返し
def policy_iter(env,gamma,threshold=0.001,is_render=False):
    #状態遷移率を集めてコネコネしてπにしてるよ～ん
    #状態専科率味のπを召し上がれ～
    pi=defaultdict(lambda:{0:0.25,1:0.25,2:0.25,3:0.25})
    #状態価値関数の初期化
    V=defaultdict(lambda:0)

    while True:
        V=policy_eval(pi,V,env,gamma,threshold)#価値関数の評価
        new_pi=greedy_policy(V,env,gamma)#改善

        if is_render:
            env.render_v(V,pi)
        if new_pi == pi: #更新チェックされたかどうか
            break
        pi=new_pi
    return pi
'''
#3x4のグリッドワールドで早速動かしてみよ～
env=GridWorld()
gamma=0.9
pi=policy_iter(env,gamma,threshold=0.00001,is_render=True)
'''
#評価と改善（方策反復法の終わり）


#価値反復法の実装_________________________________________________________________________________p.124


####それぞれの行動後の価値関数
def value_iter_onestep(V,env,gamma):
    for state in env.states():#すべての状態にアクセス
        if state==env.goal_state:#ゴールの価値関数を０に
            V[state]=0
            continue
        action_values=[]
        for action in env.actions():#すべての行動にアクセス
            next_state=env.next_state(state,action)
            r=env.reward(state,action,next_state)
            value=r+gamma*V[next_state]# 新しい価値関数　max演算子の中身
            action_values.append(value)

        V[state]=max(action_values)#最大値を取り出しVに設定
    return V

####value_iter_onestepを繰り返し呼び出す。価値関数の更新量が閾値を下回るまで繰り返す

def value_iter(V,env,gamma,threshold=0.001,is_render=True):
    while True:
        if is_render:
            env.render_v(V)
        old_V = V.copy() #更新前の価値関数
        V=value_iter_onestep(V,env,gamma)
        #更新された量の最大値
        delta=0
        for state in V.keys():
            t=abs(V[state]- old_V[state])
            if delta<t:
                delta = t
        #閾値との比較
        if delta<threshold:
            break
    return V

'''
#####実行
from policy_iter import greedy_policy

V=defaultdict(lambda:0)
env=GridWorld()
gamma=0.9

V=value_iter(V,env,gamma)
pi =greedy_policy(V,env,gamma)
env.render_v(V,pi)
'''

#価値反復法終わり_______________________________________________________________________________________________________










        

    







    

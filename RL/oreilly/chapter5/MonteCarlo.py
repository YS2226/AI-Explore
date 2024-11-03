import sys
sys.path.append("C:/Users/YS/Desktop/RL/scratch/chapter4")
from gridworld import GridWorld
#黄色のリンターがついてるが、特に問題はない

###モンテカルロの実装_______________________________________________________________________________p.143
env=GridWorld()
action = 0 #ダミーの行動
next_state,reward,done=env.step(action) 

'''step関数について:
どうやら行動を受け取って、次の状態、報酬、そして関数が走り切ったか(boolean)を出力する
Stepメソッドを使ってモンテカルロのサンプルを集める

'''

'''
print('next_state: ',next_state)
print('reward: ',reward)
print('done: ',done)
'''

###環境と状態のリセット
from collections import defaultdict
import numpy as np

env=GridWorld()
state=env.reset()

#モンテカルロを使って方策評価を行うエージェントを実装
#ここでのエージェントはランダムは方策に従って行動するものとする
class RandomAgent:
    def __init__(self):
        self.gamma=0.9
        self.action_size=4
        #確率分布
        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        #方策ππ
        self.pi = defaultdict(lambda:random_actions)
        #価値関数ストック用ディク
        self.V=defaultdict(lambda:0)
        #インクリメンタル実装を使って収益の平均を求める際に使う
        self.cnts=defaultdict(lambda:0)
        #エージェントが行動して得た経験値(状態、行動、報酬)のデータストック
        self.memory=[]
    #状態を受け取り行動を返すメソッド
    def get_action(self,state):
        #np.random.choiceを使ってランダムに行動させる
        action_probs=self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    #経験値を保存
    def add(self,state,action,reward):
        data=(state,action,reward)
        self.memory.append(data)
    #メモリーリセット
    def reset(self):
        self.memory.clear()
    #モンテカルロを実行
    def eval(self):
        #収益？
        G=0
        for data in reversed(self.memory): #逆向きにたどる。（モンテカルロの効率化のため）
            state,action,reward = data
            G=self.gamma*G + reward
            self.cnts[state] +=1
            self.V[state] +=(G - self.V[state]) / self.cnts[state]

#モンテカルロ方策評価を動かす_________________________________________

'''
env=GridWorld()
agent=RandomAgent()

episodes = 1000
for episodes in range(episodes):
    state=env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)

        #状態、行動委、報酬のサンプルデータを記録
        agent.add(state,action,reward)
        if done:
            #モンテカルロによって価値を更新
            agent.eval()
            break

        state=next_state
env.render_v(agent.V)

'''

#モンテカルロ方策評価終わり______________________________________________________________________________

#モンテカルロ方策制御はっじまるよーん＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿
"""
p.150
モンテカルロで方策制御をおこなう意味
前回の3x4のグリッドワールドで使った u(s)=argmax_a Σ_s' p(s'|s,a){r(s,a,s')+ γV(s')} 
この式は状態遷移率と報酬関数が既知の場合においてのみ使える式
しかーし、世の中そんな簡単に上手くいかないものですね、大抵の場合はわからないんですよ
そこーでぇぇぇぇ↑　下の式をつかうんですねぇ

= argmax_a Q(s,a)
これを行う場合、状態価値関数 ではなくて　Q関数に評価を行わなければなりませーぬ
つまるところ今回はQ関数に対してモンテカルロっちゃおうぜってことです　うぇーい！　うぇーい！
"""

#評価するものが違うとはいえコードは大体同じ。
class McAgent:
    def __init__(self):
        self.gamma=0.9
        #何種類の行動ができるか定義？
        self.action_size=4
        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda:random_actions)
        self.Q = defaultdict(lambda:0)
        self.cnts=defaultdict(lambda:0)
        self.memory=[]
        self.epsilon = 0.1 # 修正1より　完全なgreedyからεgreedyへ
        self.alpha = 0.1 # Q値を更新する際の固定値
    #状態を受け取り行動を返すメソッド
    def get_action(self,state):
        #np.random.choiceを使ってランダムに行動させる
        action_probs=self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    #経験値を保存
    def add(self,state,action,reward):
        data=(state,action,reward)
        self.memory.append(data)
    #メモリーリセット
    def reset(self):
        self.memory.clear()

    # このupdateメソッドと下のgreedy_probsがメインの方策制御のコード
    def update(self):
        G=0
        for data in reversed(self.memory): #逆向きにたどる。（モンテカルロの効率化のため）
            state,action,reward = data
            G=self.gamma*G + reward
            #行動価値関数を使うため、因数に状態と行動を挟む
            key = (state,action)

            #修正２
            #self.Q[key] +=(G - self.Q[key]) / self.cnts[key]
            #修正前の↑はサンプルデータに対して平均的な重みを求めていたが、ここでは指数移動平均が適しているため、固定値αを使った指数移動平均を使う。

            self.Q[key]+= (G - self.Q[key])*self.alpha #固定値

            self.pi[state] = greedy_probs(self.Q,state,self.epsilon)

#今後greedy_probsは今後他のクラスからもアクセスしたいのでクラスの外に置く
#行動の確率分布を出力する
def greedy_probs(Q,state,epsilon,action_size=4): 
    qs = [Q[(state,action)] for action in range(action_size)]
    max_action = np.argmax(qs)


    base_prob = epsilon / action_size
    action_probs={action:base_prob for action in range(action_size)}
    #この時点でaction_probsは{0:epsilon/4,1:e/4,2:e/4,3:e/4}

    #修正１
    #↓は完全なgreedyな行動だけを行うのでエージェントのたどるルートが１つになってしまう。つまりサンプルデータが欲しいのに探索しなくなるのでデータ
    ####集まらなくなるわけだ
    #action_probs[max_action]=1
    action_probs[max_action] += (1-epsilon)
    return action_probs
    """
    このメソッドの出力
    したがって、例としてεが0.2で、最適行動が action = 2 である場合、action_probs のディクショナリは以下のようになります：

    action_probs = {
        0: 0.05,
        1: 0.05,
        2: 0.85,
        3: 0.05
    }
    """

#モンテカルロ方策制御実行

env = GridWorld()
agent=McAgent()

episodes =10000
for episodes in range(episodes):
    state=env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)

        agent.add(state,action,reward)
        
        if done:
            #エピソードが終了したら収益が確定して初めて方策に修正が入る。
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)




    


    
    




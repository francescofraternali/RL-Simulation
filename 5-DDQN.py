# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#
# author: Jaromir Janisch, 2016

import random, numpy, math, sys
from keras import backend as K

import tensorflow as tf

# My stuff
from Reward_Policies_Func import Env_Rew_Pol

MEMORY_CAPACITY = 100000  # It was 100000
MIN_EPSILON = 0.1 # Default 0.01
tot_episodes = 100000; tot_action = 3

granularity = "1h" # see option above
if granularity == "1min":
    granul = [3600, 1800, 600, 300, 60]
elif granularity == "5min":
    granul = [3600, 1800, 1200, 600, 300]
elif granularity == "10min":
    granul = [7200, 3600, 1800, 1200, 600]
elif granularity == "30min":
    granul = [9000, 7200, 5400, 3600, 1800]
elif granularity == "1h":
    granul = [14400, 10800, 7200, 3600]
else:
    print("Granularity Error")
    exit()

diction_feat = {0 : 'SC_feed', 1: 'Light_feed'}
#diction_feat = {0 : 'SC_feed', 1: 'Light_feed', 2: 'Time'}
diction = {"MEMORY_CAPACITY": 100000, 100000 : "10k", 200000: "2k"}

MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]
Text = "5-DDQN_N("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + diction_feat[i] + "_"
    else:
        Text = Text + diction_feat[i]

Text = Text + ")-R30M-" + str(tot_action) + "-" + granularity
Text = Text + ")-Mem_" + diction[MEMORY_CAPACITY]

#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

#----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        #model.compile(loss=huber_loss, optimizer=opt) # Original
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
#MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
#MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

            '''
        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507]) # Original
            #S = numpy.array([0, 0, 0, 0])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()
        '''
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)
        pTarget_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN
                #t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    #def __init__(self, problem):
    #    self.problem = problem
    #    self.env = gym.make(problem)

        def run(self, agent):
            Environ = Env_Rew_Pol()
            for episode in range(tot_episodes):
                # initial observation
                s, s_ = Environ.Init(diction_feat, granul)
                #s_ = s
                #while curr_time < end_time: # here it's like the agent(BS) wakes up and sense the light and battery
                while True: # here it's like the agent(BS) wakes up and sense the light and battery

                    # First the agent (BS) takes an action (i.e. telling Pible to send data or not to send)
                    action = agent.act(s) # The BS tells Pible to increase perf with 1 while 0 is decrease perf

                    reward, done, s_ = Environ.Envi(action, tot_action, diction_feat, granul, s_)


                    agent.observe( (s, action, reward, s_) )
                    agent.replay()

                    s = Environ.Update_s(action, diction_feat, s)
                    # break while loop when end of this episode
                    if done:
                        break

                Environ.printing(Text, episode, tot_episodes)


#-------------------- MAIN ----------------------------
#PROBLEM = 'CartPole-v0'
#env = Environment(PROBLEM)
env = Environment()

#stateCnt  = env.env.observation_space.shape[0]
#actionCnt = env.env.action_space.n

stateCnt = len(diction_feat)
actionCnt = tot_action

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)


while randomAgent.memory.isFull() == False:
    env.run(randomAgent)

print "Start Learning"
exit()
agent.memory.samples = randomAgent.memory.samples
randomAgent = None
#printasfuck = 1
#for i in range(0,tot_episodes):
env.run(agent)
#    episode += 1

#Start Plotting
#print("Best Lenght", len(Time_Best))
#print("Best total Reward", best_reward)

# end of game
#print('Game Over and Best Perc Reward:', "%.2f%%" % Perc_Best, Text)
#End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
#print("Start Time: " + Start_Real_Time + ", End Time: " + End_Time)

#fig.savefig('/mnt/c/Users/Francesco/Desktop/Dropbox/EH/RL/RL_MY/Images/'+Text+'.png', bbox_inches='tight')
#rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
#rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)

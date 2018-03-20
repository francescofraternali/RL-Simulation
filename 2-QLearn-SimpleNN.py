# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
#
# author: Jaromir Janisch, 2016

#--- enable this to run on GPU
# import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import random, numpy, math
from Reward_Policies_Func import Env_Rew_Pol

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

# My stuff

#Settings

MAX_EPSILON = 1
MIN_EPSILON = 0.01 # Default 0.01

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

tot_episodes = 1000000; save_rev_rate = 10000; tot_action = 3
diction_feat = {0: 'curr_time_h_feed', 1: 'SC_norm'}

diction = {"MEMORY_CAPACITY": 100000, 100000 : "10k", 200000: "2k"}

MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]

Text = "2-QSimNN_TL_NE("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + str(diction_feat[i]) + "_"
    else:
        Text = Text + str(diction_feat[i])

Text = Text + ")-Rew30M-" + str(tot_action) + "-" + granularity

# My stuff over

class Brain: # It encapsulate the neural network. predict(s) predicts the Q function values in state s, while train(batch) performs supervised training step with batch
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):    # here we have the definition of the Neural Network with Keras Library
        model = Sequential()


        #model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))  #Original
        #model.add(Dense(output_dim=actionCnt, activation='linear'))     #Original
        model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)
            #global SampleSam
            #global episode
            #if episode not in SampleSam:
            #    SampleSam.append(episode)
            #print("Full")  #Francesco

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
#MEMORY_CAPACITY = 100000  # It was 100000
BATCH_SIZE = 64 # It was 64

GAMMA = 0.99  # Gamma is the discount factor


#MIN_EPSILON = 0.001 # Default 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):   # First thing to do: take an action
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

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
        p_ = self.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
    '''
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        print("here")
        print(self.env)
        print("over")
    '''

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

                agent.observe( (s, action, reward, s_) ) # Observe new sample
                agent.replay()  # do a leaning step

                s = Environ.Update_s(action, diction_feat, s)
                # break while loop when end of this episode
                if done:
                    break

            Environ.printing(Text, episode, tot_episodes)

#-------------------- MAIN ----------------------------
#PROBLEM = 'MountainCar-v0'
#env = Environment(PROBLEM)
env = Environment()

#stateCnt  = env.env.observation_space.shape[0]

stateCnt = len(diction_feat)
#actionCnt = env.env.action_space.n
actionCnt = tot_action

agent = Agent(stateCnt, actionCnt)

#global best_reward; global episodes; global Text
#best_reward = 0
#episode = 0

env.run(agent)
#if i % 10 == 0 and len:
#    rp.plot_legend(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best)
#finally:
#    agent.brain.model.save("cartpole-basic.h5")

# end of game
#print('Game Over and Best Perc Reward:', "%.2f%%" % Perc_Best, Text)
#End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
#print("Start Time: " + Start_Real_Time + ", End Time: " + End_Time)

#print("SampleSam:" , SampleSam)
#rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
#rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)

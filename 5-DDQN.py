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
import pickle

import tensorflow as tf

# My stuff
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Reward_Policies_Func as rp

#Settings
Start_Real_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
Starting_time = datetime.datetime(2018,1,1,12,00,00)
Ending_time = datetime.datetime(2018,1,4,12,00,00)

Init_SC_Volt = 3.5; SC_norm_max = 10; SC_norm_min = 1

reset = 60*60*24; cnt_max = 10000
Perc_Best = 0; save_rev_rate = 1000
Time_h_min = 0; Time_h_max = 24; curr_time_h_feed = 0.00; curr_time_h_feed_next = 0.00
norm_light_min = 0; norm_light_max = 1
Light_max = 10; Light_min = 0; Light_feed = 0.00
# My stuff over

MEMORY_CAPACITY = 100000  # It was 100000
MIN_EPSILON = 0.01 # Default 0.01
tot_episodes = 100000;

diction_feat = {0 : 'SC_feed', 1: 'Light_feed'}
#diction_feat = {0 : 'SC_feed', 1: 'Light_feed', 2: 'Time'}
diction = {"MEMORY_CAPACITY": 100000, 100000 : "10k", 200000: "2k"}

MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]
Text = "5-DDQN_EndT("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + diction_feat[i] + "_"
    else:
        Text = Text + diction_feat[i]

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
        # Initiallization
        global Perc_Best; global best_reward
        global episode; global Tot_Episodes
        start_time = Starting_time
        curr_time = start_time
        curr_time_next = curr_time
        end_time = Ending_time
        curr_time_h = curr_time.hour
        curr_time_h_feed = (curr_time_h - Time_h_min)/float((Time_h_max - Time_h_min))
        curr_time_h_feed = round(curr_time_h_feed,2)
        #Light_sec = 8*60*60 # 8 hour
        Light = 10; Perc = 0; cnt = 0; perf = 8; time_temp = 4; reward = 0; R = 0; done = False; Occupancy = 0
        SC_temp = Init_SC_Volt
        SC_temp, SC_norm, SC_feed = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
        cnt_hist = []; Light_hist = []; Action_hist = []; r_hist = []; Time_hist = []; perf_hist = []; SC_hist = []; SC_norm_hist = []; Occup_hist = []; Light_feed_hist = []; SC_feed_hist = []; Tot_Rew_hist = []

        # Normalize all parameters from 0 to 1 before feeding the RL
        Light_feed = (Light - Light_min)/float((Light_max - Light_min))

        #s = np.array([SC_feed, SC_feed, curr_time_h_feed, Light_feed])
        if len(diction_feat) == 1:
            s = np.array([SC_feed])
        elif len(diction_feat) == 2:
            s = np.array([SC_feed, Light_feed])
        else:
            s = np.array([SC_feed, Light_feed, curr_time_h_feed])

        s_ = s

        #while curr_time < end_time: # here it's like the agent(BS) wakes up and sense the light and battery
        while True: # here it's like the agent(BS) wakes up and sense the light and battery

            # First the agent (BS) takes an action (i.e. telling Pible to send data or not to send)
            action = agent.act(s) # The BS tells Pible to increase perf with 1 while 0 is decrease perf

            #Reward Based on Action and Environment
            #reward, perf = rp.simple_light_rew(action, Light, perf)
            reward, perf_next = rp.simple_barath_sending_and_dying_rew(action, SC_norm, perf, SC_norm_min)
            #reward, perf = rp.all_rew(action, Light, SC_norm, SC_norm_old, perf, curr_time_h, Occupancy)

            # Adjust Performance and Time
            perf_next, time_temp = rp.adjust_perf(perf_next)
            #time_temp = 600

            # Environment starts here
            # Environment changes based on the action taken, here here I Calculate Next Light intensity and Occupancy
            Light_next, Occupancy_next, Light_feed_next = rp.calc_light_occup(curr_time_h)

            # Calculate Energy Produced and Consumed Based on the action taken
            SC_temp_next, SC_norm_next, SC_feed_next = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
            # Adjust Environment Time
            curr_time_next = curr_time + datetime.timedelta(0,time_temp)
            curr_time_h_next = curr_time_next.hour
            curr_time_h_feed_next = (curr_time_h_next - Time_h_min)/float((Time_h_max - Time_h_min))
            curr_time_h_feed_next = round(curr_time_h_feed_next,2)

            cnt += 1;
            #reward += 1
            R += reward

            #if cnt >= cnt_max:
            if curr_time >= end_time:
                done = True

            # Update s
            #s_[0] = SC_feed_next; s_[1] = SC_feed_next; s_[2] = curr_time_h_feed_next; s_[3] = Light_feed_next;

            # Update s
            if len(diction_feat) == 1:
                s_[0] = SC_feed_next
            elif len(diction_feat) == 2:
                s_[0] = SC_feed_next; s_[1] = Light_feed_next
            else:
                s_[0] = SC_feed_next; s_[1] = Light_feed_next; s_[2] = curr_time_h_feed_next; #s_[3] = curr_time_h;

            #s_ = np.array([Light_feed, SC_feed])
            # End Environment, now the Agent Observe an Learn from it

            if done: # terminal state
                s_ = None

            agent.observe( (s, action, reward, s_) )
            agent.replay()

            # Append Data
            Light_hist.append(Light); Action_hist.append(action); cnt_hist.append(cnt); r_hist.append(reward); Time_hist.append(curr_time); perf_hist.append(perf); SC_hist.append(SC_temp); SC_norm_hist.append(SC_norm); Occup_hist.append(Occupancy); Light_feed_hist.append(Light_feed); SC_feed_hist.append(SC_feed);

            # Swap observation
            s = s_
            perf = perf_next; Light = Light_next; curr_time = curr_time_next; curr_time_h = curr_time_h_next; Occupancy = Occupancy_next
            SC_norm = SC_norm_next; SC_temp = SC_temp_next; Light_feed = Light_feed_next; SC_feed = SC_feed_next

            # break while loop when end of this episode
            if done:
                break

        Perc = (100 * float(R)/float(len(Time_hist)))
        print(Text + ", Epis: " + str(episode) + "/" + str(tot_episodes) + ", Rew: " + str(R) + ", Max_R: " + str(best_reward) + ", Started: " + Start_Real_Time)
        Tot_Reward.append(R); Tot_Episodes.append(episode)
        if R > best_reward:
            global Time_Best; global Light_Best; global cnt_Best; global SC_Best; global perf_Best; global Action_Best; global r_Best; global SC_Best_norm_hist; global SC_Feed_Best; global Light_Feed_Best; global Occup_Best
            Light_Best = Light_hist; Action_Best = Action_hist; r_Best = r_hist; cnt_Best = cnt_hist; best_reward = R; Time_Best = Time_hist
            perf_Best = perf_hist; SC_Best = SC_hist; SC_Best_norm_hist = SC_norm_hist; Perc_Best = Perc; Occup_Best = Occup_hist; Light_Feed_Best = Light_feed_hist; SC_Feed_Best = SC_feed_hist;
            rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, episode)
            rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, episode)
            with open('Saved_Data/' + Text + '.pkl', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([Light_Best, Action_Best, r_Best, cnt_Best, best_reward, Time_Best, perf_Best, SC_Best, SC_Best_norm_hist, Perc_Best, Occup_Best, Light_Feed_Best, SC_Feed_Best, Tot_Episodes, Tot_Reward, Text, episode], f)

        if episode % save_rev_rate == 0:
            rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, episode)


#-------------------- MAIN ----------------------------
#PROBLEM = 'CartPole-v0'
#env = Environment(PROBLEM)
env = Environment()

#stateCnt  = env.env.observation_space.shape[0]
#actionCnt = env.env.action_space.n
global Time_Best; global Light_Best; global cnt_Best; global SC_Best; global perf_Best; global Action_Best; global r_Best; global SC_Best_norm_hist; global SC_Feed_Best; global Light_Feed_Best; global Occup_Best; global Tot_Reward; global Tot_Episodes;
global episode; global best_reward
Light_Best = []; SC_Best = []; perf_Best = []; Time_Best = []; Action_Best = []; r_Best = []; cnt_Best = []; SC_Best_norm_hist = []; SC_Feed_Best = []; Light_Feed_Best = []; Occup_Best = []; Tot_Reward = []; Tot_Episodes = []
best_reward = 0
episode = 0

printasfuck  = 0
stateCnt = len(diction_feat)
actionCnt = 3

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)


while randomAgent.memory.isFull() == False:
    env.run(randomAgent)

agent.memory.samples = randomAgent.memory.samples
randomAgent = None
printasfuck = 1
for i in range(0,tot_episodes):
    env.run(agent)
    episode += 1

#Start Plotting
print("Best Lenght", len(Time_Best))
print("Best total Reward", best_reward)

# end of game
print('Game Over and Best Perc Reward:', "%.2f%%" % Perc_Best, Text)
End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
print("Start Time: " + Start_Real_Time + ", End Time: " + End_Time)

#fig.savefig('/mnt/c/Users/Francesco/Desktop/Dropbox/EH/RL/RL_MY/Images/'+Text+'.png', bbox_inches='tight')
rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)

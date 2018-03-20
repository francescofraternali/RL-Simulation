# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
#
# author: Jaromir Janisch, 2016

import random, numpy, math, scipy
from SumTree import SumTree

#IMAGE_WIDTH = 84
#IMAGE_HEIGHT = 84
#IMAGE_STACK = 2

# My stuff
from Reward_Policies_Func import Env_Rew_Pol

MEMORY_CAPACITY = 1000000  # It was 200000
MIN_EPSILON = 0.1 # Default 0.01
tot_episodes = 500000; tot_action = 3

granularity = "1min" # see option above
if granularity == "1min":
    granul = [3600, 1800, 600, 60]
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

diction_feat = {0 : 'curr_time_h', 1 : 'curr_time_m',  2: 'Light', 3: 'SC_norm'}
#diction_feat = {0 : 'SC_feed', 1: 'Light_feed', 2: 'Time'}
diction = {"MEMORY_CAPACITY": 200000, 100000 : "10k", 200000: "20k", 500000: "50k", 1000000: "1M"}

MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]
Text = "6-DDQN_PER_N("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + diction_feat[i] + "_"
    else:
        Text = Text + diction_feat[i]

Text = Text + ")-Rew30M-" + str(tot_action) + "-" + granularity
Text = Text + ")-Mem_" + diction[MEMORY_CAPACITY]

# My stuff over

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00025

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

'''
def processImage( img ):
    rgb = scipy.misc.imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    return o
'''

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()

        #model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        #model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(Flatten())

        model.add(Dense(units=512, activation='relu', input_dim=stateCnt)) #Francesco it was 64
        #model.add(Dense(units=actionCnt, activation='linear'))

        #model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        #return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
#MEMORY_CAPACITY = 200000

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        #x = numpy.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

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
#PROBLEM = 'Seaquest-v0'
#env = Environment(PROBLEM)
env = Environment()

#stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
stateCnt = len(diction_feat)
#actionCnt = env.env.action_space.n
actionCnt = tot_action

#print(stateCnt)
#print(actionCnt)

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

#try:
print("Initialization with random agent...")
'''
while randomAgent.exp < MEMORY_CAPACITY:
    env.run(randomAgent)
    print(randomAgent.exp, "/", MEMORY_CAPACITY)
'''
agent.memory = randomAgent.memory

randomAgent = None

print("Starting learning")

env.run(agent)

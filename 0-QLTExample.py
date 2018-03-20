"""
Reinforcement learning.
The RL is in RL_brain.py.

One cool thing to do: put a pattern of the presence of people and check if it can predict that
"""

from RL_brain import QLearningTable
import numpy as np
from random import *
from Reward_Policies_Func import Env_Rew_Pol
import time

print("Remember to export DISPLAY=:0")


# Try to give him all the power by letting him decide which perf to take over time

granul = [0,1,2,3,4,5]

tot_episodes = 1000000; tot_action = 6

Text = "Merda"


# My stuff over

def update():
    Environ = Env_Rew_Pol()
    for episode in range(tot_episodes):

        R = [[-1, -1, -1, -1, 0, -1],
                [-1, -1, -1, 0, -1, 100],
                [-1, -1, -1, 0, -1, -1],
                [-1, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, -1, 100],
                [-1, 0, -1, -1, 0, 100]]

        #s_ = s
        s = 1
        s_ = 1
        done = False
        #while curr_time < end_time:
        while True:

            # RL choose action based on observation
            #action = RL.choose_action(str(s))
            action = randint(0, tot_action-1)
            while True:
                if R[s][action] == -1:
                    action = randint(0, tot_action-1)
                else:
                    break

            if s_ == 5:
                done = True

            s_ = action

            reward = R[s][action]

            #print "reward: " + str(reward)

            #print("before", s, action, reward, s_)
            # RL learn from this transition
            #print ("s, action, reward, s_", s, action, reward, s_)
            RL.learn(str(s), action, reward, str(s_), Text, tot_action, len(granul))

            #time.sleep(3)
            #print(s, action, reward, s_)

            s = s_            # break while loop when end of this episode

            if done == True:
                break

        #print("Episode Over")

        # end of game
    '''
    print('Game Over, Best Reward Ever:', "%.2f%%" % Perc_Best, Text)
    End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
    print("Started Time: " + Start_Real_Time + ", End Time: " + End_Time)
    rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
    rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)
    '''

if __name__ == "__main__":
    #env = Maze()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(tot_action)))
    update()
    #env.after(100, update)
    #env.mainloop()

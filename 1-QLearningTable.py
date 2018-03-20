"""
Reinforcement learning.
The RL is in RL_brain.py.

One cool thing to do: put a pattern of the presence of people and check if it can predict that
"""

from RL_brain import QLearningTable
import numpy as np
from Reward_Policies_Func import Env_Rew_Pol
import time

print("Remember to export DISPLAY=:0")


# Try to give him all the power by letting him decide which perf to take over time

granularity = "5min" # see option above
if granularity == "1min":
    granul = [3600, 1800, 600, 300, 60]
elif granularity == "5min":
    granul = [3600, 1800, 1200, 600, 300]
elif granularity == "10min":
    granul = [3600, 1800, 1200, 600]
elif granularity == "30min":
    granul = [5400, 3600, 1800]
elif granularity == "1h":
    granul = [10800, 7200, 3600]
else:
    print "Granularity Error"
    exit()

tot_episodes = 1000000; tot_action = 3
diction_feat = {0: 'curr_time_h', 1:'curr_time_m', 2:'SC_norm', 3: 'Light'}

#diction_feat = {SC_feed: 0}
#print "Value : %d" %  diction_feat.get(SC_feed)

#diction = {"MEMORY_CAPACITY": 100000, 100000 : "10k", 200000: "2k"}

#MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]

Text = "1-Qtable_TL_NE("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + str(diction_feat[i]) + "_"
    else:
        Text = Text + str(diction_feat[i])

Text = Text + ")-R3G-" + str(tot_action) + "-" + granularity

# My stuff over

def update():
    Environ = Env_Rew_Pol()
    for episode in range(tot_episodes):

        #s_ = s
        s, s_ = Environ.Init(diction_feat, granul)
        #while curr_time < end_time:
        while True:

            # RL choose action based on observation
            action = RL.choose_action(str(s))

            reward, done, s_ = Environ.Envi(action, tot_action, diction_feat, granul, s_)

            #print("before", s, action, reward, s_)
            #time.sleep(10)
            # RL learn from this transition

            RL.learn(str(s), action, reward, str(s_), Text, tot_action, granul)

            #print(s, action, reward, s_)

            s = Environ.Update_s(action, diction_feat, s, granul)

            # break while loop when end of this episode
            if done:
                break

        Environ.printing(Text, episode, tot_episodes)

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

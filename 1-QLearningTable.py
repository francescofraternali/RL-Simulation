"""
Reinforcement learning.
The RL is in RL_brain.py.

One cool thing to do: put a pattern of the presence of people and check if it can predict that
"""

from RL_brain import QLearningTable
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Reward_Policies_Func as rp
import pickle

print("Remember to export DISPLAY=:0")

#Settings
Start_Real_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
Starting_time = datetime.datetime(2018,1,1,12,00,00)
Ending_time = datetime.datetime(2018,1,8,7,00,00)

Init_SC_Volt = 3.5; SC_norm_max = 10; SC_norm_min = 1

#I_sleep = 0.0000015; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3; V_Solar = 1.5; I_Solar = I_Solar_200lux; SC_size = 1.5
Light_Best = []; SC_Best = []; perf_Best = []; Time_Best = []; Action_Best = []; r_Best = []; cnt_Best = []; SC_Best_norm_hist = []; SC_Feed_Best = []; Light_Feed_Best = []; Tot_Reward = []; Tot_Episodes = []
reset = 60*60*24; cnt_max = 10000
best_reward = 0; Perc_Best = 0
#on_off =  Look at the Reward_Policies_Func
norm_light_min = 0; norm_light_max = 1
Light_max = 10; Light_min = 0; Light_feed = 0.00
Time_h_min = 0; Time_h_max = 24; curr_time_h_feed = 0.00; curr_time_h_feed_next = 0.00

tot_episodes = 1000000; save_rev_rate = 10000; tot_action = 3

global diction_feat
diction_feat = {0: 'curr_time_h', 1: 'SC_norm', 2: 'Light'}

#diction_feat = {SC_feed: 0}
#print "Value : %d" %  diction_feat.get(SC_feed)

#diction = {"MEMORY_CAPACITY": 100000, 100000 : "10k", 200000: "2k"}

#MEMORY_CAPACITY = diction["MEMORY_CAPACITY"]

#Text = "QSimpleNN(" + diction[0] + "-" + diction[1] + ")-Mem_" + diction[MEMORY_CAPACITY]

Text = "1-Qtable_TL_NE-Simple("
for i in range(0,len(diction_feat)):
    if i < len(diction_feat)-1:
        Text = Text + str(diction_feat[i]) + "_"
    else:
        Text = Text + str(diction_feat[i])

Text = Text + ")-Act" + str(tot_action)

# My stuff over

def update():
    best_reward = 0
    for episode in range(tot_episodes):
        # initial observation
        start_time = Starting_time
        curr_time = start_time
        curr_time_next = curr_time
        end_time = Ending_time
        curr_time_h = curr_time.hour
        curr_time_m = curr_time_h * 60 + curr_time.minute
        #Light_sec = 8*60*60 # 8 hour
        Light = 10; Perc = 0; cnt = 0; perf = 8; time_temp = 4; reward = 0; R = 0; done = False; Occupancy = 0
        SC_temp = Init_SC_Volt
        SC_real = Init_SC_Volt
        SC_temp, SC_norm, SC_feed = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
        SC_real = round(SC_temp,1)
        cnt_hist = []; Light_hist = []; Action_hist = []; r_hist = []; Time_hist = []; perf_hist = []; SC_hist = []; SC_norm_hist = []; Occup_hist = []; Light_feed_hist = []; SC_feed_hist = []

        # Normalize all parameters from 0 to 1 before feeding the RL
        Light_feed = (Light - Light_min)/float((Light_max - Light_min))
        curr_time_h_feed = (curr_time_h - Time_h_min)/float((Time_h_max - Time_h_min))

        stuff = []
        for i in range(len(diction_feat)):
            tocheck = diction_feat[i]
            feature = rp.checker(tocheck, SC_norm, SC_feed, SC_temp, SC_real, Light, Light_feed, curr_time_h, curr_time_m)
            stuff.append(feature)


        if len(diction_feat) == 1:
            s = np.array([stuff[0]]); s_ = np.array([stuff[0]])
        elif len(diction_feat) == 2:
            s = np.array([stuff[0], stuff[1]]); s_ = np.array([stuff[0], stuff[1]])
        elif len(diction_feat) == 3:
            s = np.array([stuff[0], stuff[1], stuff[2]]); s_ = np.array([stuff[0], stuff[1], stuff[2]])
        elif len(diction_feat) == 4:
            s = np.array([stuff[0], stuff[1], stuff[2], stuff[3]]); s_ = np.array([stuff[0], stuff[1], stuff[2], stuff[3]])
        else:
            print("Dictionary Feature Error")
            exit()

        #s_ = s

        #while curr_time < end_time:
        while True:

            # RL choose action based on observation
            action = RL.choose_action(str(s))

            #print(s)

            # Reward Based on Action and Environment
            reward, perf_next = rp.simple_barath_sending_and_dying_rew(action, tot_action, SC_norm, perf, SC_norm_min)
            #reward, perf_next = rp.simple_light_rew(action, Light, perf)
            #reward, perf_next = rp.simple_time_rew(action, Light, perf)

            # Adjust Performance and Time
            perf_next, time_temp = rp.adjust_perf(perf_next)  #time_temp = 600

            # Environment starts here
            # Adjust Environment Time
            curr_time_next = curr_time + datetime.timedelta(0,time_temp)
            curr_time_h_next = curr_time_next.hour
            curr_time_m_next = curr_time_h_next * 60 + curr_time_next.minute
            curr_time_h_feed_next = (curr_time_h_next - Time_h_min)/float((Time_h_max - Time_h_min))
            curr_time_h_feed_next = round(curr_time_h_feed_next,2)

            # Environment changes based on the action taken, here I Calculate Next Light intensity and Occupancy
            Light_next, Occupancy_next, Light_feed_next = rp.calc_light_occup(curr_time_h_next)

            # Calculate Energy Produced and Consumed Based on the action taken
            SC_temp_next, SC_norm_next, SC_feed_next = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
            SC_real_next = round(SC_temp_next,1)

            cnt += 1;

            R += reward

            #if cnt >= cnt_max:
            if curr_time >= end_time:
                done = True

            for i in range(len(diction_feat)):
                tocheck = diction_feat[i]
                feature = rp.checker(tocheck, SC_norm_next, SC_feed_next, SC_temp_next, SC_real_next, Light_next, Light_feed_next, curr_time_h_next, curr_time_m_next)
                s_[i] = feature

            #s_[0] = curr_time_h_next; s_[1] = Light_next; s_[2] = SC_temp_next;

            '''
            if len(diction_feat) == 1:
                s_[0] = SC_norm_next;
            elif len(diction_feat) == 2:
                s_[0] = SC_norm_next; s_[1] = Light_next;
            else:
                s_[0] = SC_norm_next; s_[1] = Light_next; s_[2] = curr_time_h_next; #s_[3] = curr_time_h;
            #observation_ = [SC_feed_next, Light_feed_next, curr_time_h_feed_next]
            '''
            #print("before", s, action, reward, s_)
            # RL learn from this transition
            RL.learn(str(s), action, reward, str(s_))

            #print(s, action, reward, s_)

            # Append Data
            Light_hist.append(Light); Action_hist.append(action); cnt_hist.append(cnt); r_hist.append(reward); Time_hist.append(curr_time); perf_hist.append(perf); SC_hist.append(SC_temp); SC_norm_hist.append(SC_norm); Occup_hist.append(Occupancy); Light_feed_hist.append(Light_feed); SC_feed_hist.append(SC_feed)

            # swap observation
            for i in range(len(diction_feat)):
                tocheck = diction_feat[i]
                feature = rp.checker(tocheck, SC_norm_next, SC_feed_next, SC_temp_next, SC_real_next, Light_next, Light_feed_next, curr_time_h_next, curr_time_m_next)
                s[i] = feature
            #s = s_;
            perf = perf_next; Light = Light_next; curr_time = curr_time_next; curr_time_h = curr_time_h_next; Occupancy = Occupancy_next; SC_norm = SC_norm_next; SC_temp = SC_temp_next; Light_feed = Light_feed_next; SC_feed = SC_feed_next; curr_time_m = curr_time_m_next;

            # break while loop when end of this episode
            if done:
                break

        Perc = (100 * float(R)/float(len(Time_hist)))
        global Perc_Best; global Tot_Reward; global Tot_Episodes
        print(Text + ", Epis: " + str(episode) + "/" + str(tot_episodes) + ", Rew: " + str(R) + ", Max_R: " + str(best_reward) + ", Started: " + Start_Real_Time)
        if R < 0:
            R = 0
        Tot_Reward.append(R); Tot_Episodes.append(episode)
        episode += 1

        if R > best_reward:
            Light_Best = Light_hist; Action_Best = Action_hist; r_Best = r_hist; cnt_Best = cnt_hist; best_reward = R; Time_Best = Time_hist; perf_Best = perf_hist; SC_Best = SC_hist
            SC_Best_norm_hist = SC_norm_hist; Perc_Best = Perc; Occup_Best = Occup_hist; Light_Feed_Best = Light_feed_hist; SC_Feed_Best = SC_feed_hist

            rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, episode)
            rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, episode)
            with open('Saved_Data/' + Text + '.pkl', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([Light_Best, Action_Best, r_Best, cnt_Best, best_reward, Time_Best, perf_Best, SC_Best, SC_Best_norm_hist, Perc_Best, Occup_Best, Light_Feed_Best, SC_Feed_Best, Tot_Episodes, Tot_Reward, Text, episode], f)

        if episode % save_rev_rate == 0:
            rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, episode)

        # end of game
    print('Game Over, Best Reward Ever:', "%.2f%%" % Perc_Best, Text)
    End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
    print("Started Time: " + Start_Real_Time + ", End Time: " + End_Time)
    rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
    rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)


if __name__ == "__main__":
    #env = Maze()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(tot_action)))
    update()
    #env.after(100, update)
    #env.mainloop()

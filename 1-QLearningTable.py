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

print("Remember to export DISPLAY=:0")

#Settings
Start_Real_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
Starting_time = datetime.datetime(2018,1,1,12,00,00)
Ending_time = datetime.datetime(2018,1,4,7,00,00)

Init_SC_Volt = 3.5; SC_norm_max = 10; SC_norm_min = 1

#I_sleep = 0.0000015; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3; V_Solar = 1.5; I_Solar = I_Solar_200lux; SC_size = 1.5
Light_Best = []; SC_Best = []; perf_Best = []; Time_Best = []; Action_Best = []; r_Best = []; cnt_Best = []; SC_Best_norm_hist = []; SC_Feed_Best = []; Light_Feed_Best = []; Tot_Reward = []; Tot_Episodes = []
reset = 60*60*24; cnt_max = 10000
best_reward = 0; Perc_Best = 0
#on_off =  Look at the Reward_Policies_Func
norm_light_min = 0; norm_light_max = 1
Light_max = 10; Light_min = 0; Light_feed = 0.00
Time_h_min = 0; Time_h_max = 24; curr_time_h_feed = 0.00; curr_time_h_feed_next = 0.00

tot_episodes = 2000; Text = "Q-Table(SC, Light, Time)"

def update():
    best_reward = 0
    for episode in range(tot_episodes):
        # initial observation
        start_time = Starting_time
        curr_time = start_time
        curr_time_next = curr_time
        end_time = Ending_time
        curr_time_h = curr_time.hour
        #Light_sec = 8*60*60 # 8 hour
        Light = 10; Perc = 0; cnt = 0; perf = 8; time_temp = 4; reward = 0; R = 0; done = False; Occupancy = 0
        SC_temp = Init_SC_Volt
        #SC_norm_old = 0
        SC_temp, SC_norm, SC_feed = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
        cnt_hist = []; Light_hist = []; Action_hist = []; r_hist = []; Time_hist = []; perf_hist = []; SC_hist = []; SC_norm_hist = []; Occup_hist = []; Light_feed_hist = []; SC_feed_hist = []

        # Normalize all parameters from 0 to 1 before feeding the RL
        Light_feed = (Light - Light_min)/float((Light_max - Light_min))
        curr_time_h_feed = (curr_time_h - Time_h_min)/float((Time_h_max - Time_h_min))

        observation = [SC_feed, Light_feed, curr_time_h_feed]
        observation_ = observation

        #while curr_time < end_time:
        while True:

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            #Reward Based on Action and Environment
            reward, perf_next = rp.simple_barath_sending_and_dying_rew(action, SC_norm, perf, SC_norm_min)

            # Adjust Performance and Time
            perf_next, time_temp = rp.adjust_perf(perf_next)  #time_temp = 600

            # Environment starts here # Environment changes based on the action taken, here I Calculate Next Light intensity and Occupancy
            Light_next, Occupancy_next, Light_feed_next = rp.calc_light_occup(curr_time_h)

            # Calculate Energy Produced and Consumed Based on the action taken
            SC_temp_next, SC_norm_next, SC_feed_next = rp.calc_energy_prod_consu(time_temp, SC_temp, Light)
            # Adjust Environment Time
            curr_time_next = curr_time + datetime.timedelta(0,time_temp)
            curr_time_h_next = curr_time_next.hour
            curr_time_h_feed_next = (curr_time_h_next - Time_h_min)/float((Time_h_max - Time_h_min))
            curr_time_h_feed_next = round(curr_time_h_feed_next,2)
            #print(curr_time_h_feed_next)
            #print(curr_time_h_next)

            cnt += 1;
            #reward += 1
            R += reward

            if cnt >= cnt_max:
                done = True

            observation_ = [SC_feed_next, Light_feed_next, curr_time_h_feed_next]

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # Append Data
            Light_hist.append(Light); Action_hist.append(action); cnt_hist.append(cnt); r_hist.append(reward); Time_hist.append(curr_time); perf_hist.append(perf); SC_hist.append(SC_temp); SC_norm_hist.append(SC_norm); Occup_hist.append(Occupancy); Light_feed_hist.append(Light_feed); SC_feed_hist.append(SC_feed)

            # swap observation
            observation = observation_; perf = perf_next; Light = Light_next; curr_time = curr_time_next; curr_time_h = curr_time_h_next; Occupancy = Occupancy_next; SC_norm = SC_norm_next; SC_temp = SC_temp_next; Light_feed = Light_feed_next; SC_feed = SC_feed_next

            # break while loop when end of this episode
            if done:
                break

        Perc = (100 * float(R)/float(len(Time_hist)))
        global Perc_Best; global Tot_Reward; global Tot_Episodes
        print(Text + ", Epis: " + str(episode) + "/" + str(tot_episodes) + ", Rew: " + str(R) + ", Max_R: " + str(best_reward) + ", Started: " + Start_Real_Time)
        Tot_Reward.append(R); Tot_Episodes.append(episode)
        episode += 1

        if R > best_reward:
            Light_Best = Light_hist; Action_Best = Action_hist; r_Best = r_hist; cnt_Best = cnt_hist; best_reward = R; Time_Best = Time_hist; perf_Best = perf_hist; SC_Best = SC_hist
            SC_Best_norm_hist = SC_norm_hist; Perc_Best = Perc; Occup_Best = Occup_hist; Light_Feed_Best = Light_feed_hist; SC_Feed_Best = SC_feed_hist

        # end of game
    print('Game Over, Best Reward Ever:', "%.2f%%" % Perc_Best, Text)
    End_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
    print("Started Time: " + Start_Real_Time + ", End Time: " + End_Time)
    rp.plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes)
    rp.plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes)


if __name__ == "__main__":
    #env = Maze()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(3)))
    update()
    #env.after(100, update)
    #env.mainloop()

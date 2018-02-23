"""
Reinforcement learning.
The RL is in RL_brain.py.
"""

#from maze_env import Maze
from RL_brain import QLearningTable
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Settings
Starting_time = datetime.datetime(2018,1,1,7,00,00)
Ending_time = datetime.datetime(2018,1,2,10,00,00)

I_Solar_200lux = 0.000031
SC_Volt_max = 5.50; SC_Volt_min = 2.1; Init_SC_Volt = 3.5; I_sleep = 0.0000015; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3; V_Solar = 1.5; I_Solar = I_Solar_200lux; SC_size = 1.5
Light_Best = []; SC_Best = []; perf_Best = []; Time_Best = []; Action_Best = []; r_Best = []; cnt_Best = []; SC_Best_norm_hist = []
reset = 60*60*24
best_reward = 0; Perc_Best = 0

def update():
    best_reward = 0
    for episode in range(100):
        # initial observation
        #observation = env.reset()
        start_time = Starting_time
        curr_time = start_time
        end_time = Ending_time
        curr_time_h = curr_time.hour
        #Light_sec = 8*60*60 # 8 hour
        Light = 1
        Perc = 0
        cnt = 0
        perf = 4
        time_temp = 4
        SC_temp = Init_SC_Volt
        reward = 0
        R = 0
        done = False
        cnt_hist = []; Light_hist = []; Action_hist = []; r_hist = []; Time_hist = []; perf_hist = []; SC_hist = []; SC_norm_hist = []

        observation = [Light]
        observation_ = observation
        #print("Observation: ", observation)

        while curr_time < end_time:
            reward = 0
            # fresh env;
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            #print("Action: ", action)

            # RL takes action and get next observation and reward

            #observation_, reward, done = env.step(action)
            if curr_time_h >= 7 and curr_time_h < 9:
                Light = 2
            if curr_time_h >= 9 and curr_time_h < 11:
                Light = 5
            if curr_time_h >= 11 and curr_time_h < 14:
                Light = 10
            if curr_time_h >= 14 and curr_time_h < 16:
                Light = 3
            else:
                Light = 0

            # Calculate Energy Produced and Consumed
            Energy_Used = ((time_temp - Time_BLE_Sens_1)*SC_temp*I_sleep) + (Time_BLE_Sens_1*SC_temp*I_BLE_Sens_1)
            Volt_Used = round(np.sqrt(Energy_Used/(0.5*SC_size)),2)
            Energy_Prod = time_temp * V_Solar * I_Solar * Light
            Volt_Prod = round(np.sqrt(Energy_Prod/(0.5*SC_size)),2)
            SC_temp = SC_temp + Volt_Prod - Volt_Used
            if SC_temp > SC_Volt_max:
                SC_temp = SC_Volt_max
                perf = 8
            # Normalize SC_Volt from 2.1 % 5.5 to 0 % 10
            SC_norm = (10 - 0)/(SC_Volt_max-SC_Volt_min)*(SC_temp - SC_Volt_max) + 10
            SC_norm = int(round(SC_norm))

            #Reward Based on Action
            if action == 1 and Light > 0:
                reward = 1; perf += 1
            elif action == 0 and Light == 0:
                reward = 1; perf -= 1
            else:
                reward = 0

            # Adjust Performance
            if perf > 8:
            	perf = 8
            elif perf < 1:
            	perf = 1

            if perf == 8:
                time_temp = 20
            elif perf == 7:
                time_temp = 40
            elif perf == 6:
                 time_temp = 60
            elif perf == 5:
                time_temp = 120
            elif perf == 4:
                time_temp = 300
            elif perf == 3:
                time_temp = 600
            elif perf == 2:
                time_temp = 1800
            elif perf == 1:
                time_temp = 3600
            else:
                time_temp = 3600

            # Adjust Environment Parameters
            curr_time = curr_time + datetime.timedelta(0,time_temp)
            #Light_sec -= time_temp
            cnt += 1;
            R += reward
            #time_temp = 600

            # Append Data
            Light_hist.append(Light)
            Action_hist.append(action)
            cnt_hist.append(cnt)
            r_hist.append(reward)
            Time_hist.append(curr_time)
            perf_hist.append(perf)
            SC_hist.append(SC_temp)
            SC_norm_hist.append(SC_norm)

            ''''
            if curr_time >= start_time + datetime.timedelta(0, 3600):
                Light_sec = 8*60*60
            if curr_time >= start_time + datetime.timedelta(0, 7200):
                Light_sec = 0
            if curr_time >= start_time + datetime.timedelta(0, 10800):
                Light_sec = 0

            if curr_time >= start_time + datetime.timedelta(0, reset):
                Light_sec = 8*60*60
                start_time = curr_time
            '''
            if done:
                break

            if SC_temp < SC_Volt_min:
                reward = - 100
                done = True

            #print("Observation_Out", observation_)
            #print("Done: ", done)
            observation_ = [Light]

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode

            if done:
                break

        Perc = (100 * float(R)/float(len(Time_hist)))
        global Perc_Best
        if Perc > Perc_Best:
            Light_Best = Light_hist
            Action_Best = Action_hist
            r_Best = r_hist
            cnt_Best = cnt_hist
            best_reward = R
            Time_Best = Time_hist
            perf_Best = perf_hist
            SC_Best = SC_hist
            SC_Best_norm_hist = SC_norm_hist
            Perc_Best = Perc

            #Start Plotting
            fig, ax = plt.subplots(1)
            fig.autofmt_xdate()
            #plt.plot(Time_Best, SC_Volt_Best, 'ro', Time_Best, perf_Best, 'b', Time_Best, Light_Best,'c.', Time_Best, Action_Best, 'y*', Time_Best, r_Best,'k')
            plt.plot(Time_Best, Light_Best, 'b', label = 'Light' )
            plt.plot(Time_Best, Action_Best, 'y*', label = 'Action')
            plt.plot(Time_Best, r_Best, 'k+', label = 'Reward')
            #plt.plot(Time_Best, perf_Best, 'g', label = 'Performance')
            plt.plot(Time_Best, SC_Best, 'r+', label = 'SC_Voltage')
            xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)
            plt.title('Temporary Best Experimental Result')
            plt.ylabel('Super Capacitor Voltage[V]')
            plt.xlabel('Time[h]')
            plt.show()

        print("Episode", episode, "Total Number of Points", len(Time_hist), "Reward", R, "Percentage","%.2f%%" % Perc)

    # end of game
    print('Game Over and Best Percentage Reward Ever:', "%.2f%%" % Perc_Best)
    #print('Best Reward Ever: ', Perc_Best)
    #Start Plotting
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    #plt.plot(Time_Best, SC_Volt_Best, 'ro', Time_Best, perf_Best, 'b', Time_Best, Light_Best,'c.', Time_Best, Action_Best, 'y*', Time_Best, r_Best,'k')
    plt.plot(Time_Best, Light_Best, 'b', label = 'Light' )
    plt.plot(Time_Best, Action_Best, 'y*', label = 'Action')
    plt.plot(Time_Best, r_Best, 'k+', label = 'Reward')
    #plt.plot(Time_Best, perf_Best, 'g', label = 'Performance')
    plt.plot(Time_Best, SC_Best, 'r+', label = 'SC_Voltage')
    #plt.plot(Time_Best, SC_Best_norm_hist, 'm^', label = 'SC_Voltage_Normalized')
    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Best Experimental Result', fontsize=15)
    plt.ylabel('Super Capacitor Voltage[V]', fontsize=15)
    plt.xlabel('Time[h]', fontsize=20)
    ax.grid(True)
    fig.savefig('/mnt/c/Users/Francesco/Desktop/Testtest.png', bbox_inches='tight')
    plt.show()

    #savefig('/mnt/c/Users/Francesco/Desktop/Testtest.png')
    #env.destroy()

if __name__ == "__main__":
    #env = Maze()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(2)))
    update()
    #env.after(100, update)
    #env.mainloop()

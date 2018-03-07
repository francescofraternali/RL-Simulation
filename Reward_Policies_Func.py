'''
Reward Policies for Pible Simulation

'''
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

I_sleep = 0.0000015; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3
V_Solar_200lux = 1.5; I_Solar_200lux = 0.000031   # It was 1.5
SC_Volt_min = 2.1; SC_Volt_max = 5.5; SC_size = 1.5
on_off = 0 # 0 is off and it uses 10 light levels. With 1 light is just on or off
Light_max = 10; Light_min = 0

def simple_light_rew(action, Light, perf):
    if action == 1 and Light > 0:
        reward = 1; perf += 1
    elif action == 0 and Light == 0:
        reward = 1; perf -= 1
    else:
        reward = 0

    return reward, perf

def simple_barath_sending_and_dying_rew(action, SC_norm, perf, SC_norm_min): # not finished yet

    if action == 2:
        perf += 1;
    if action == 1:
        perf += 0
    if action == 0:
        perf -= 1

    if perf > 10:
    	perf = 10
    if perf < 1:
    	perf = 1

    reward = perf

    if SC_norm <= SC_norm_min:
        reward = - 3000

    return reward, perf

def adjust_perf(perf):

    if perf > 10:
    	perf = 10
    if perf < 1:
    	perf = 1

    if perf == 10:
        time_temp = 10
    elif perf == 9:
        time_temp = 20
    elif perf == 8:
        time_temp = 40
    elif perf == 7:
         time_temp = 60
    elif perf == 6:
        time_temp = 120
    elif perf == 5:
        time_temp = 300
    elif perf == 4:
        time_temp = 600
    elif perf == 3:
        time_temp = 1800
    elif perf == 2:
        time_temp = 3600
    elif perf == 1:
        time_temp = 7200
    else:
        time_temp = 7200

    return perf, time_temp

def calc_light_occup(curr_time_h):

    if curr_time_h >= 7 and curr_time_h < 8:
        Light = 2; Occupancy = 0
    elif curr_time_h >= 8 and curr_time_h < 9:
        Light = 4; Occupancy = 1
    elif curr_time_h >= 9 and curr_time_h < 10:
        Light = 6; Occupancy = 1
    elif curr_time_h >= 10 and curr_time_h < 11:
        Light = 7; Occupancy = 1
    elif curr_time_h >= 11 and curr_time_h < 12:
        Light = 9; Occupancy = 1
    elif curr_time_h >= 12 and curr_time_h < 13:
        Light = 10; Occupancy = 1
    elif curr_time_h >= 13 and curr_time_h < 15:
        Light = 8; Occupancy = 1
    elif curr_time_h >= 15 and curr_time_h < 16:
        Light = 6; Occupancy = 1
    elif curr_time_h >= 16 and curr_time_h < 18:
        Light = 3; Occupancy = 1
    else:
        Light = 0; Occupancy = 0

    if on_off == 1:
        if curr_time_h >=7 and curr_time_h < 17:
            Light = 10; Occupancy = 1
        else:
            Light = 0; Occupancy = 0

    Light_feed_next = (Light - Light_min)/float((Light_max - Light_min))
    #Light_feed_next = round(Light_feed_next, 2)

    return Light, Occupancy, Light_feed_next

def calc_energy_prod_consu(time_temp, SC_temp, Light):

    Energy_Used = ((time_temp - Time_BLE_Sens_1) * SC_temp * I_sleep) + (Time_BLE_Sens_1 * SC_temp * I_BLE_Sens_1)
    #Volt_Used = round(np.sqrt((2*Energy_Used)/SC_size),2)
    Volt_Used = np.sqrt((2*Energy_Used)/SC_size)
    Energy_Prod = time_temp * V_Solar_200lux * I_Solar_200lux * Light
    #Volt_Prod = round(np.sqrt((2*Energy_Prod)/SC_size),2)
    Volt_Prod = np.sqrt((2*Energy_Prod)/SC_size)
    SC_temp = SC_temp + Volt_Prod - Volt_Used
    if SC_temp > SC_Volt_max:
        SC_temp = SC_Volt_max

    if SC_temp < SC_Volt_min:
        SC_temp = SC_Volt_min

    # Normalize SC_Volt from 2.1 % 5.5 to 0 % 10
    SC_norm = (10 - 0)/(SC_Volt_max - SC_Volt_min)*(SC_temp - SC_Volt_max) + 10
    SC_norm = int(round(SC_norm))

    SC_feed_next = (1 - 0)/(SC_Volt_max - SC_Volt_min)*(SC_temp - SC_Volt_max) + 1
    SC_feed_next = round(SC_feed_next, 2)

    return SC_temp, SC_norm, SC_feed_next

def plot_reward_text(Tot_Episodes, Tot_Reward, Text, best_reward, tot_episodes):
    #Start Plotting
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(Tot_Episodes, Tot_Reward, 'r', label = 'Total Reward')
    #xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Reward Result - ' + Text + 'Best_R: ' + str(best_reward) + 'Epis: ' + str(tot_episodes), fontsize=15)
    plt.ylabel('Total Reward [num]', fontsize=15)
    plt.xlabel('Episode [num]', fontsize=20)
    ax.grid(True)
    fig.savefig('/mnt/c/Users/Francesco/Desktop/Dropbox/EH/RL/RL_MY/Images-Auto/Reward_' + Text + '.png', bbox_inches='tight')
    plt.show()

def plot_legend_text(Time_Best, Light_Best, Light_Feed_Best, Action_Best, r_Best, perf_Best, SC_Best, SC_Best_norm_hist, SC_Feed_Best, Occup_Best, Text, best_reward, tot_episodes):
    #Start Plotting
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(Time_Best, Light_Best, 'b', label = 'Light Feeded')
    #plt.plot(Time_Best, Light_Feed_Best, 'b', label = 'Light Feeded')
    plt.plot(Time_Best, Action_Best, 'y*', label = 'Action',  markersize = 15)
    plt.plot(Time_Best, r_Best, 'k+', label = 'Reward')
    plt.plot(Time_Best, perf_Best, 'g', label = 'Performance')
    plt.plot(Time_Best, SC_Best, 'r+', label = 'SC_Voltage')
    plt.plot(Time_Best, SC_Best_norm_hist, 'm^', label = 'SC_Voltage_Normalized')
    plt.plot(Time_Best, SC_Best_norm_hist, 'm')
    #plt.plot(Time_Best, SC_Feed_Best, 'c^', label = 'SC_Voltage_Feeded')
    #plt.plot(Time_Best, Occup_Best, 'c^', label = 'Occupancy')
    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Best Experimental Result - ' + Text + 'Best_R: ' + str(best_reward) + 'Epis: ' + str(tot_episodes), fontsize=15)
    plt.ylabel('Super Capacitor Voltage[V]', fontsize=15)
    plt.xlabel('Time[h]', fontsize=20)
    ax.grid(True)
    fig.savefig('/mnt/c/Users/Francesco/Desktop/Dropbox/EH/RL/RL_MY/Images-Auto/' + Text + '.png', bbox_inches='tight')
    plt.show()

'''
Reward Policies for Pible Simulation

'''
import numpy as np
import datetime
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pickle

I_sleep = 0.0000015; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3
V_Solar_200lux = 0.9; I_Solar_200lux = 0.000007   # It was 1.5
SC_Volt_min = 2.1; SC_Volt_max = 5.5; SC_size = 1
on_off = 0 # 0 is off and it uses 10 light levels. With 1 light is just on or off
Light_max = 10; Light_min = 0;
save_rev_rate = 100000;

class Env_Rew_Pol:
    def __init__(self):
        #Settings Env Variable
        self.Start_Real_Time = datetime.datetime.now().strftime('%m-%d %H:%M')
        self.Starting_time = datetime.datetime(2018,1,1,6,00,00)
        self.Ending_time = datetime.datetime(2018,1,2,6,00,00)
        self.Init_SC_Volt = 3; self.SC_norm_max = 10; self.SC_norm_min = 0
        self.Light_feed = 0.0
        self.Time_h_min = 0; self.Time_h_max = 24; self.Time_m_max = 60
        self.curr_time_h_feed = 0.0; self.curr_time_h_feed_next = 0.0
        self.best_reward = 0; self.Perc_Best = 0
        self.Light_Best = []; self.SC_Best = []; self.perf_Best = []; self.Time_Best = []; self.Action_Best = []; self.r_Best = []; self.cnt_Best = []; self.SC_Best_norm_hist = []; self.SC_Feed_Best = []; self.Light_Feed_Best = [];
        self.Tot_Reward = []; self.Tot_Episodes = []

    def Init(self, diction_feat, granul):
        # initial observation
        self.curr_time = self.Starting_time
        self.end_time = self.Ending_time
        self.curr_time_h = self.curr_time.hour
        self.curr_time_m = self.curr_time.minute
        self.curr_time_h_feed = (self.curr_time_h - self.Time_h_min)/float((self.Time_h_max - self.Time_h_min))
        self.curr_time_m_feed = (self.curr_time_m - 0)/float((self.Time_m_max - 0))
        self.Light = 0; self.perf = int(len(granul)/2); self.time_temp = 4; self.reward = 0; self.R = 0; self.done = False; self.Occupancy = 0

        self.SC_real = self.Init_SC_Volt
        self.SC_temp, self.SC_norm, self.SC_feed = calc_energy_prod_consu(self.time_temp, self.Init_SC_Volt, self.Light)
        self.SC_real = round(self.SC_temp,1)
        self.cnt_hist = []; self.Light_hist = []; self.Action_hist = []; self.r_hist = []; self.Time_hist = []; self.perf_hist = []; self.SC_hist = []; self.SC_norm_hist = []; self.Occup_hist = []; self.Light_feed_hist = []; self.SC_feed_hist = []

        # Normalize all parameters from 0 to 1 before feeding the RL
        self.Light_feed = (self.Light - Light_min)/float((Light_max - Light_min))

        stuff = []
        for i in range(len(diction_feat)):
            tocheck = diction_feat[i]
            feature = checker(tocheck, self.SC_norm, self.SC_feed, self.SC_temp, self.SC_real, self.Light, self.Light_feed, self.curr_time_h, self.curr_time_m, self.curr_time_h_feed, self.curr_time_m_feed)
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

        return s, s_

    def Envi(self, action, tot_action, diction_feat, granul, s_):
        # Reward Based on Action and Environment
        self.reward, self.perf_next = simple_barath_sending_and_dying_rew(action, tot_action, self.SC_norm, self.perf, self.SC_norm_min, granul)
        #reward, perf_next = rp.simple_light_rew(action, Light, perf)

        # Adjust Performance and Time
        self.perf_next, self.time_temp = adjust_perf(self.perf_next, granul)  #time_temp = 600

        # Environment starts here
        # Adjust Environment Time
        self.curr_time_next = self.curr_time + datetime.timedelta(0, self.time_temp)
        self.curr_time_h_next = self.curr_time_next.hour
        self.curr_time_m_next = self.curr_time_next.minute
        self.curr_time_h_feed_next = (self.curr_time_h_next - self.Time_h_min)/float((self.Time_h_max - self.Time_h_min))
        self.curr_time_h_feed_next = round(self.curr_time_h_feed_next,2)
        self.curr_time_m_feed_next = (self.curr_time_m_next - 0)/float((self.Time_m_max - 0))

        # Environment changes based on the action taken, here I Calculate Next Light intensity and Occupancy
        self.Light_next, self.Occupancy_next, self.Light_feed_next = calc_light_occup(self.curr_time_h_next)

        # Calculate Energy Produced and Consumed Based on the action taken
        self.SC_temp_next, self.SC_norm_next, self.SC_feed_next = calc_energy_prod_consu(self.time_temp, self.SC_temp, self.Light)
        self.SC_real_next = round(self.SC_temp_next,1)

        self.cnt = 1;

        self.R += self.reward

        #if cnt >= cnt_max:
        if self.curr_time >= self.end_time:
            self.done = True

        for i in range(len(diction_feat)):
            tocheck = diction_feat[i]
            feature = checker(tocheck, self.SC_norm_next, self.SC_feed_next, self.SC_temp_next, self.SC_real_next, self.Light_next, self.Light_feed_next, self.curr_time_h_next, self.curr_time_m_next, self.curr_time_h_feed_next, self.curr_time_m_feed_next)
            s_[i] = feature

        return self.reward, self.done, s_

    def Update_s(self, action, diction_feat, s, granul):
        # Append Data
        self.Light_hist.append(self.Light); self.Action_hist.append(action); self.cnt_hist.append(self.cnt); self.r_hist.append(self.reward/((len(granul)-1)*100)); self.Time_hist.append(self.curr_time); self.perf_hist.append(self.perf/1000); self.SC_hist.append(self.SC_temp); self.SC_norm_hist.append(self.SC_norm); self.Occup_hist.append(self.Occupancy);
        self.Light_feed_hist.append(self.Light_feed); self.SC_feed_hist.append(self.SC_feed)

        # swap observation
        for i in range(len(diction_feat)):
            tocheck = diction_feat[i]
            feature = checker(tocheck, self.SC_norm_next, self.SC_feed_next, self.SC_temp_next, self.SC_real_next, self.Light_next, self.Light_feed_next, self.curr_time_h_next, self.curr_time_m_next, self.curr_time_h_feed_next, self.curr_time_m_feed_next)
            s[i] = feature
        #s = s_;
        self.perf = self.perf_next;
        self.Light = self.Light_next; self.Light_feed = self.Light_feed_next;
        self.curr_time = self.curr_time_next; self.curr_time_h = self.curr_time_h_next; self.curr_time_m = self.curr_time_m_next;
        self.Occupancy = self.Occupancy_next;
        self.SC_norm = self.SC_norm_next; self.SC_temp = self.SC_temp_next; self.SC_feed = self.SC_feed_next;

        return s

    def printing(self, Text, episode, tot_episodes):

        self.Perc = (100 * float(self.R)/float(len(self.Time_hist)))
        print(Text + ", Epis: " + str(episode) + "/" + str(tot_episodes) + ", Rew: " + str(self.R) + ", Max_R: " + str(self.best_reward) + ", Started: " + self.Start_Real_Time)
        if self.R < 0:
            self.R = 0
        self.Tot_Reward.append(self.R); self.Tot_Episodes.append(episode)

        if self.R > self.best_reward:
            self.Light_Best = self.Light_hist; self.Action_Best = self.Action_hist; self.r_Best = self.r_hist; self.cnt_Best = self.cnt_hist; self.best_reward = self.R; self.Time_Best = self.Time_hist; self.perf_Best = self.perf_hist; self.SC_Best = self.SC_hist
            self.SC_Best_norm_hist = self.SC_norm_hist; self.Perc_Best = self.Perc; self.Occup_Best = self.Occup_hist; self.Light_Feed_Best = self.Light_feed_hist; self.SC_Feed_Best = self.SC_feed_hist

            plot_legend_text(self.Time_Best, self.Light_Best, self.Light_Feed_Best, self.Action_Best, self.r_Best, self.perf_Best, self.SC_Best, self.SC_Best_norm_hist, self.SC_Feed_Best, self.Occup_Best, Text, self.best_reward, episode)
            plot_reward_text(self.Tot_Episodes, self.Tot_Reward, Text, self.best_reward, episode)
            with open('Saved_Data/' + Text + '.pkl', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.Light_Best, self.Action_Best, self.r_Best, self.cnt_Best, self.best_reward, self.Time_Best, self.perf_Best, self.SC_Best, self.SC_Best_norm_hist, self.Perc_Best, self.Occup_Best, self.Light_Feed_Best, self.SC_Feed_Best, self.Tot_Episodes, self.Tot_Reward, Text, episode], f)

        if episode % save_rev_rate == 0:
            plot_reward_text(self.Tot_Episodes, self.Tot_Reward, Text, self.best_reward, episode)




def simple_light_rew(action, Light, perf):
    if action == 2 and Light > 0:
        reward = 1; perf += 1
    elif action == 0 and Light == 0:
        reward = 1; perf -= 1
    else:
        reward = 0

    perf = 7

    return reward, perf

def simple_barath_sending_and_dying_rew(action, tot_action, SC_norm, perf, SC_norm_min, granul): # not finished yet

    if tot_action == 3:
        if action == 2:
            perf += 1;
        if action == 1:
            perf += 0
        if action == 0:
            perf -= 1
    elif tot_action == 2:
        if action == 1:
            perf += 1
        if action == 0:
            perf -= 1
    else:
        print("Error")
        exit()

    if perf > len(granul)-1:
    	perf = len(granul)-1
    if perf < 0:
    	perf = 0

    reward = perf * 100

    if SC_norm <= SC_norm_min:
        reward = - 3000000

    return reward, perf

def adjust_perf(perf, granul):

    if perf > len(granul) - 1:
    	perf = len(granul) - 1
    if perf < 0:
    	perf = 0

    time_temp = granul[perf]

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

    SC_energy = SC_temp * SC_temp * 0.5 * SC_size
    Energy_Used = ((time_temp - Time_BLE_Sens_1) * SC_temp * I_sleep) + (Time_BLE_Sens_1 * SC_temp * I_BLE_Sens_1)
    #Volt_Used = round(np.sqrt((2*Energy_Used)/SC_size),2)
    Volt_Used = np.sqrt((2*Energy_Used)/SC_size)
    Energy_Prod = time_temp * V_Solar_200lux * I_Solar_200lux * Light
    #Volt_Prod = round(np.sqrt((2*Energy_Prod)/SC_size),2)
    Volt_Prod = np.sqrt((2*Energy_Prod)/SC_size)
    #print("SC_temp_old, time_temp, Volt_Prod, Volt_Used", SC_temp, time_temp, Volt_Prod, Volt_Used)
    SC_temp = SC_temp + Volt_Prod - Volt_Used
    #print("SC_temp_final", SC_temp)

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
    #fig.autofmt_xdate()
    plt.plot(Tot_Episodes, Tot_Reward, 'r', label = 'Total Reward')
    #xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Reward Trend - ' + Text + ', Best_R: ' + str(best_reward) + ', Epis: ' + str(tot_episodes), fontsize=15)
    plt.ylabel('Total Reward [num]', fontsize=15)
    plt.xlabel('Episode [num]', fontsize=20)
    ax.grid(True)
    #fig.savefig('/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Images-Auto/Reward_' + Text + '.png', bbox_inches='tight')
    fig.savefig('Images-Auto/Reward_' + Text + '.png', bbox_inches='tight')
    #plt.show()
    plt.close(fig)

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
    plt.title(Text + ', Best_R: ' + str(best_reward) + ', Epis: ' + str(tot_episodes), fontsize=15)
    plt.ylabel('Super Capacitor Voltage[V]', fontsize=15)
    plt.xlabel('Time[h]', fontsize=20)
    ax.grid(True)
    fig.savefig('Images-Auto/' + Text + '.png', bbox_inches='tight')
    #plt.show()
    plt.close(fig)


def checker(tc, SC_norm, SC_feed, SC_temp, SC_real, Light, Light_feed, curr_time_h, curr_time_m, curr_time_h_feed, curr_time_m_feed):
    if tc == "SC_norm":
        return SC_norm
    elif tc == "Light":
        return Light
    elif tc == "SC_temp":
        return SC_temp
    elif tc == "SC_real":
        return SC_real
    elif tc == "SC_feed":
        return SC_feed
    elif tc == "Light_feed":
        return Light_feed
    elif tc == "curr_time_h":
        return curr_time_h
    elif tc == "curr_time_m":
        return curr_time_m
    elif tc == "curr_time_h_feed":
        return curr_time_h_feed
    elif tc == "curr_time_m_feed":
        return curr_time_m_feed

    else:
        print("Checher Error")
        return "error"

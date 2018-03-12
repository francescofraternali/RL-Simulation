import pickle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

Text = 'DDQN_PER_EndT(SC_feed_Light_feed)-Mem_20k'
with open(Text + '.pkl') as f:  # Python 3: open(..., 'rb')
    Light_Best, Action_Best, r_Best, cnt_Best, best_reward, Time_Best, perf_Best, SC_Best, SC_Best_norm_hist, Perc_Best, Occup_Best, Light_Feed_Best, SC_Feed_Best = pickle.load(f)


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
#fig.savefig('/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Images-Auto/Reward_' + Text + '.png', bbox_inches='tight')
plt.show()
plt.close(fig)


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
#fig.savefig('/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Images-Auto/' + Text + '.png', bbox_inches='tight')
plt.show()
plt.close(fig)

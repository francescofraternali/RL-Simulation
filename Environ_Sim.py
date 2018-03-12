import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

I_sleep = 0.0000045; I_BLE_Sens_1 = 0.000199; Time_BLE_Sens_1 = 3
V_Solar_200lux = 1; I_Solar_200lux = 0.000011   # It was 1.5
SC_Volt_min = 2.1; SC_Volt_max = 5.5; SC_size = 1

SC_temp = 3.5
time_temp = 60
Light = 4
curr_time = datetime.datetime(2018,1,1,12,00,00)
Time = []; Time.append(curr_time)
SC_hist = []; SC_hist.append(SC_temp)


while True:
    Energy_Used = ((time_temp - Time_BLE_Sens_1) * SC_temp * I_sleep) + (Time_BLE_Sens_1 * SC_temp * I_BLE_Sens_1)
    #Volt_Used = round(np.sqrt((2*Energy_Used)/SC_size),2)
    Volt_Used = np.sqrt((2*Energy_Used)/SC_size)
    Energy_Prod = time_temp * V_Solar_200lux * I_Solar_200lux * Light
    print(Energy_Prod)
    #Volt_Prod = round(np.sqrt((2*Energy_Prod)/SC_size),2)
    Volt_Prod = np.sqrt((2*Energy_Prod)/SC_size)
    print("SC_temp_old, time_temp, V_Prod, V_Used", SC_temp, time_temp, Volt_Prod, Volt_Used)
    SC_temp = SC_temp + Volt_Prod - Volt_Used
    #SC_Perc = (SC_energy * 100)/Energy_Full
    print("SC_temp_final", SC_temp)

    if SC_temp > SC_Volt_max:
        SC_temp = SC_Volt_max

    if SC_temp < SC_Volt_min:
        SC_temp = SC_Volt_min

    curr_time = curr_time + datetime.timedelta(0,time_temp)

    Time.append(curr_time); SC_hist.append(SC_temp);

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(Time, SC_hist, 'r+', label = 'SC_Voltage')
    plt.plot(Time, SC_hist, 'r')
    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Environment Simulation', fontsize=15)
    plt.ylabel('Super Capacitor Voltage[V]', fontsize=15)
    plt.xlabel('Time[h]', fontsize=20)
    ax.grid(True)
    plt.show()
    plt.close(fig)

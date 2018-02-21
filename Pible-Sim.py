#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

Dur = input('Duration of the Simulation [day]: ')
Dur_sec = Dur*60*60*24
Light = input('Light per day[h]: ')
Light_sec = Light*60*60
#print('Total Duration[h]', duration)

perf = 1

start_time = datetime.datetime(2017,1,1,7,00,00)
curr_time = start_time
end_time = curr_time + datetime.timedelta(0, Dur_sec) #days, seconds, then other fields.
reset = 60*60*24
print curr_time
print end_time

Init_SC_Volt = 3.5
SC_Volt_max = 5.5
SC_Volt_min = 2.1

SC_Volt = []
SC_Volt.append(Init_SC_Volt)

Time = []
Time.append(curr_time)

def Pible_Brain(perf, Light_sec, SC_temp):
	# Check Light
	if Light_sec > 0:
		perf += 1
                if SC_temp < SC_Volt_max:
                        SC_Volt.append(SC_temp + 0.005)
                else:
                        SC_Volt.append(SC_temp)
        else:
		perf -= 1
                if  SC_temp >= SC_Volt_min:
                        SC_Volt.append(SC_temp - 0.04)
                else:
                        SC_Volt.append(SC_temp)
	#Check SC
	if len(SC_Volt) > 5:
		if SC_Volt[-1] > SC_Volt[-2]:
			perf += 1
		else:
			perf -= 1

	if perf > 7:
		perf = 7
	elif perf < 1:
		perf = 1

	return perf


while curr_time < end_time:

	SC_temp = SC_Volt[-1]

	# Pible Brain
	perf = Pible_Brain(perf, Light_sec, SC_temp)

	# Update Perf
	if perf == 7:
	        curr_time = curr_time + datetime.timedelta(0,20)
		Light_sec -= 20
	elif perf == 6:
	        curr_time = curr_time + datetime.timedelta(0,40)
                Light_sec -= 40
	elif perf == 5:
	        curr_time = curr_time + datetime.timedelta(0,60)
                Light_sec -= 60
        elif perf == 4:
                curr_time = curr_time + datetime.timedelta(0,120)
                Light_sec -= 120
        elif perf == 3:
                curr_time = curr_time + datetime.timedelta(0,300)
                Light_sec -= 300
        elif perf == 2:
                curr_time = curr_time + datetime.timedelta(0,600)
                Light_sec -= 600
        elif perf == 1:
                curr_time = curr_time + datetime.timedelta(0,3600)
                Light_sec -= 3600


	# Update Simulation Values
        Time.append(curr_time)
	Light_sec -= 1
	if curr_time >= start_time + datetime.timedelta(0, reset):
		Light_sec = Light*60*60
		start_time = curr_time

#Start Plotting

fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(Time,SC_Volt, 'ro')
xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel('Super Capacitor Voltage[V]')
plt.xlabel('Time[h]')
plt.show()

quit()

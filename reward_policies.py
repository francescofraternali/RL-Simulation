'''
Reward Policies for Pible Simulation

'''

def simple_light_rew(action, Light, perf):
    if action == 1 and Light > 0:
        reward = 1; perf += 1
    elif action == 0 and Light == 0:
        reward = 1; perf -= 1
    else:
        reward = 0

    return reward, perf

def simple_batt_rew(action, SC_norm, perf):
    if action == 1 and SC_norm > 5:
        reward = 1; perf += 1
    elif action == 0 and SC_norm < 5:
        reward = 1; perf -= 1
    else:
        reward = 0

    return reward, perf

def simple_barath_sending_and_dying_rew(action, SC_norm, perf, SC_norm_min):
    if action == 1:
        reward = 1; perf += 1
    elif action == 0:
        reward
    else:
        reward = 0

    if SC_norm < SC_norm_min:
        reward = - 1

    return reward, perf

def all_rew(action, Light, SC_norm, perf, curr_time_h):

    if curr_time_h >=8 and curr_time_h < 12 and perf < 3:
        perf = 3

    if action == 1 and Light > 3:
        reward = 1; perf += 1
    elif action == 0 and Light <= 3:
        reward = 1; perf -= 1
    else:
        reward = 0

    return reward, perf

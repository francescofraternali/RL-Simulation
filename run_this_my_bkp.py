"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

#from maze_env import Maze
from RL_brain import QLearningTable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SC_Volt_max = 5.5; SC_Volt_min = 2.1
Light_Best = []; SC_Volt_Best = []; perf_Best = []; Time_Best = []; Action_Best = []; r_Best = []; cnt_Best = []

def update():
    best_reward = 0
    for episode in range(1000):
        # initial observation
        #observation = env.reset()
        start_time = datetime.datetime(2017,1,1,7,00,00)
        curr_time = start_time
        Light_sec = 100 # 8 hour
        Light = 1
        cnt = 0
        reward = 0
        R = 0
        done = False
        cnt_hist = []; Light_hist = []; Action_hist = []; r_hist = []; Time_hist = []

        observation = [Light]
        observation_ = observation
        #print("Observation: ", observation)

        while True:
            reward = 0
            # fresh env;
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            #print("Action: ", action)

            # RL take action and get next observation and reward

            #observation_, reward, done = env.step(action)
            Light = 1 if Light_sec > 0 else 0

            if action == 1 and Light == 1:
                reward = 1
            elif action == 0 and Light == 0:
                reward = 1
            else:
                reward = 0

            #print("reward hist before", r_hist)
            cnt += 1
            Light_sec -= 1
            Light_hist.append(Light)
            Action_hist.append(action)
            cnt_hist.append(cnt)
            r_hist.append(reward)

            R += reward
            #print("Reward", reward)
            #print("reward hist", r_hist)

            if cnt > 499 and cnt < 501:
                Light_sec = 100

            if cnt == 1000:
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

        if R > best_reward:
            Light_Best = Light_hist
            Action_Best = Action_hist
            r_Best = r_hist
            cnt_Best = cnt_hist
            best_reward = R
            '''
            #Start Plotting
            fig, ax = plt.subplots(1)
            fig.autofmt_xdate()
            #plt.plot(Time_Best, SC_Volt_Best, 'ro', Time_Best, perf_Best, 'b', Time_Best, Light_Best,'c.', Time_Best, Action_Best, 'y*', Time_Best, r_Best,'k')
            plt.plot(cnt_Best, Light_Best, 'ro', cnt_Best, Action_Best, 'y*', cnt_Best, r_Best,'k')
            #xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
            #ax.xaxis.set_major_formatter(xfmt)
            plt.title('Temporary Best Experimental Result')
            plt.ylabel('Super Capacitor Voltage[V]')
            plt.xlabel('Time[h]')
            plt.show()
            '''
        print("Episode", episode, "Best_reward", best_reward)

    # end of game
    print('Game Over and Best Reward: ', best_reward)
    print("%.0f%%" % (100 * best_reward/len(cnt_Best)))
    #Start Plotting
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    #plt.plot(Time_Best, SC_Volt_Best, 'ro', Time_Best, perf_Best, 'b', Time_Best, Light_Best,'c.', Time_Best, Action_Best, 'y*', Time_Best, r_Best,'k')
    plt.plot(cnt_Best, Light_Best, 'b', cnt_Best, Action_Best, 'y*', cnt_Best, r_Best, 'k+' )
    ax.plot(Light_Best, 'b', label='Light')
    ax.plot(Action_Best, 'y*', label='Action')
    ax.plot(r_Best, 'k+', label='Reward')
    legend = ax.legend(loc='center right', shadow=True)
    #xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    plt.title('Best Experimental Result')
    plt.ylabel('Super Capacitor Voltage[V]')
    plt.xlabel('Time[h]')
    plt.show()
    #env.destroy()

if __name__ == "__main__":
    best_reward = 0
    #env = Maze()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(2)))
    update()
    #env.after(100, update)
    #env.mainloop()

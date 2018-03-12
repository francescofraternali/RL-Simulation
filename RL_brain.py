import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.99, e_greedy=0.9):
        self.Q_table_tot = []
        self.steps_count = []
        self.steps = 0
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        #print s,a, q_predict
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        self.steps += 1
        if self.steps % 10000 == 0:
                row, column = self.q_table.shape
                print row
                print column
                print np.mean(self.q_table)

                self.Q_table_tot.append(np.mean(self.q_table))
                self.steps_count.append(self.steps/10000)
                #for i in range(0,row)
                #    for ii in range(0,column)
                #        odosods = 1
                #Start Plotting
                fig, ax = plt.subplots(1)
                #fig.autofmt_xdate()
                plt.plot(self.steps_count, self.Q_table_tot, 'r', label = 'Q function')
                ax.tick_params(axis='both', which='major', labelsize=10)
                #legend = ax.legend(loc='center right', shadow=True)
                #plt.legend(loc=9, prop={'size': 10})
                plt.title('Q function mean for 3 actions', fontsize=15)
                plt.ylabel('Q predict', fontsize=15)
                plt.xlabel('steps', fontsize=20)
                ax.grid(True)
                fig.savefig('Images-Auto/Qfunc_Qtable.png', bbox_inches='tight')
                #plt.show()
                plt.close(fig)
                #if self.steps/10000 == 1000:
                #    self.epsilon = 0.99

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            print state
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

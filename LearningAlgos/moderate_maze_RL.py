"""

"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def set_prior_qtable(self, df_qtable):
        self.q_table = df_qtable

    def set_greedy_rule(self, epoch_to_update, greedy_rate):
        self.epoch_to_update = epoch_to_update
        self.greedy_rate = greedy_rate

    def update_episode(self, episode):
        if episode != 0 and episode % self.epoch_to_update == 0:
            self.epsilon = 1 - (1 - self.epsilon)*self.greedy_rate
            # print(self.epsilon)

    def choose_action(self, observation):
        # print()
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return int(action)

    def learn(self, s, a, r, s_, is_done):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if not is_done:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
)
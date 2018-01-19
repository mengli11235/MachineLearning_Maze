"""

"""

import numpy as np
import pandas as pd


class QLearningTable:

    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.global_e_greedy = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.greedy_dict = {}
        self.agent_extra_state = ""

    def set_prior_qtable(self, df_qtable):
        self.q_table = df_qtable

    def set_greedy_rule(self, epoch_to_update, greedy_rate):
        self.epoch_to_update = epoch_to_update
        self.greedy_rate = greedy_rate

    def update_episode(self, key):
        if key not in self.greedy_dict:
            self.greedy_dict[key] = [1, self.global_e_greedy]
        else:
            obj = self.greedy_dict[key]
            epi = obj[0] + 1
            epsilon = obj[1]
            if epi != 0 and epi % self.epoch_to_update == 0:
                print(key)
                print(epi)
                print(epsilon)
                print()
                epsilon = 1 - (1 - epsilon) * self.greedy_rate
                # print(epsilon)
            self.greedy_dict[key] = [epi, epsilon]

    def choose_action(self, state):
        extra_state = str(state[2:4])
        # print()
        observation = str(state)
        self.check_state_exist(observation)
        epsilon = self.global_e_greedy
        if extra_state in self.greedy_dict:
            epsilon = self.greedy_dict[extra_state][1]
        # action selection
        if np.random.uniform() < epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return int(action)

    def learn(self, _s, a, r, _s_, is_done):
        extra_state = str(_s[2:4])
        is_virtual_done = False
        if extra_state != self.agent_extra_state:
            # future extra_state not [0, 0]
            if not(_s_[2] == 0 and _s_[3] == 0):
                is_virtual_done = True
            self.agent_extra_state = extra_state
            self.update_episode(extra_state)
        s = str(_s)
        s_ = str(_s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if not is_done:
            if is_virtual_done:
                q_target = r  # next state is like new maze
            else:
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
import numpy as np
import pandas as pd
import math


class VTable:
    
    def __init__(self, learning_rate_v, reward_decay, lambda_v):

        self.v = pd.DataFrame(columns=list(range(1)), dtype=np.float64)   # matrix of state v-values
        self.lr = learning_rate_v
        self.gamma = reward_decay
        self.lambda_v = lambda_v
        self.traces = pd.DataFrame(columns=list(range(1)), dtype=np.float64)              # matrix of eligibility traces

    def set_prior_qtable(self, df_vtable):
        self.traces = df_vtable


    def update(self, s, r, s_, is_done):
        self.check_state_exist(s)
        self.check_state_exist(s_)
        # update eligibility traces
        self.traces.ix[s, 0] = self.traces.ix[s, 0] + 1
        self.traces.ix[s_, 0] = self.gamma * self.lambda_v * self.traces.ix[s, 0]
        if not is_done:
            target_v = r + self.gamma * self.v.ix[s_, 0] - self.v.ix[s, 0]
        else:
            target_v = r - self.v.ix[s, 0]
        self.v.ix[s, 0] = self.v.ix[s, 0] + self.lr * target_v * self.traces.ix[s, 0]
        self.traces.ix[s, 0] = 0
        return self

    def check_state_exist(self, state):
        if state not in self.traces.index:
            # append new state to traces table
            self.traces = self.traces.append(
                pd.Series(
                    [0]*1,
                    index=self.traces.columns,
                    name=state,
                )
            )

        if state not in self.v.index:
            # append new state to v table
            self.v = self.v.append(
                pd.Series(
                    [0] * 1,
                    index=self.v.columns,
                    name=state,
                )
            )


class QTable:

    def __init__(self, actions, learning_rate, reward_decay, e_greedy, max_reward_coefficient):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.diff_epsilon = 0
        self.max_reward = {}
        self.max_reward_coefficient = 0.999 if max_reward_coefficient >= 1 else max_reward_coefficient
        self.agent_extra_state = ""

    def set_prior_qtable(self, df_qtable):
        self.q_table = df_qtable

    def choose_action(self, observation):
        # print()
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() <= self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return int(action)

    def learn(self, v_table, s, a, r, s_, is_done, force_exit):
        extra_state = str(s[2:4])
        extra_newstate = str(s_[2:4])
        reward_coefficient = 1
        virtual_done = False
        if is_done or force_exit:
            self.agent_extra_state = ""
        elif extra_newstate != self.agent_extra_state:
            if extra_newstate != "[0 0]":
                virtual_done = True
            # self.update_episode(extra_newstate)
            self.agent_extra_state = extra_newstate


        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        v_table.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]

        if virtual_done:
            next_expectation = 0 if extra_newstate not in self.max_reward else self.max_reward[extra_newstate]
            q_target = r + next_expectation
            reward_coefficient = self.check_max_reward(extra_state, q_target)
            # print(self.max_reward)
            # print(q_target)
        elif force_exit or (not is_done):
            q_target = r + self.gamma * v_table.v.ix[s_, 0]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            reward_coefficient = self.check_max_reward(extra_state, q_target)
            # print(q_target)

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)*reward_coefficient  # update

    def check_max_reward(self, state_key, r):
        # print(self.max_reward)
        # print(r)
        # print()
        if r <= 0:
            return 1

        if state_key not in self.max_reward:
            self.max_reward[state_key] = r
            return 1
        else:
            max_val = self.max_reward[state_key]
            max_reward_coefficient = self.max_reward_coefficient
            if max_val > r:
                std_unit = max_reward_coefficient * (1 - max_reward_coefficient) * max_val
                se = (r - max_val * max_reward_coefficient) / std_unit
                return se
            else:
                self.max_reward[state_key] = r
                return 1

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

    def update_episode(self, epi):
        if epi != 0:
            if self.epsilon <= 0.99:
                self.epsilon = self.epsilon + self.diff_epsilon
            else:
                self.epsilon = 0.99
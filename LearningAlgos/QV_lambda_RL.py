import numpy as np
import pandas as pd


class VTable:
    
    def __init__(self, max_steps, learning_rate_v, reward_decay, lambda_v):

        self.v = pd.DataFrame(columns=list(range(1)), dtype=np.float64)   # matrix of state v-values
        self.lr = learning_rate_v
        self.gamma = reward_decay
        self.lambda_v = lambda_v
        self.max_steps = max_steps
        self.traces = pd.DataFrame(columns=list(range(self.max_steps)), dtype=np.float64)              # matrix of eligibility traces
    
    def update(self, s, r, s_, t, is_done):
        self.check_state_exist(s)
        self.check_state_exist(s_)
        # update eligibility traces
        if t > 1:
            self.traces.ix[:, t] = self.gamma * self.lambda_v * self.traces.ix[:, t-1]
        self.traces.ix[s, t] = self.traces.ix[s, t] + 1
        if not is_done:
            target_v = r + self.gamma * self.v.ix[s_, 0] - self.v.ix[s, 0]
        else:
            target_v = r - self.v.ix[s, 0]
        self.v.ix[s, 0] = self.v.ix[s, 0] + self.lr * target_v * self.traces.ix[s, t]
        return self

    def check_state_exist(self, state):
        if state not in self.traces.index:
            # append new state to traces table
            self.traces = self.traces.append(
                pd.Series(
                    [0]*self.max_steps,
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

    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def set_prior_qtable(self, df_qtable):
        self.q_table = df_qtable

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

    def learn(self, v_table, s, a, r, s_, is_done):
        self.check_state_exist(s_)
        v_table.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if not is_done:
            q_target = r + self.gamma * v_table.v.ix[s_, 0]  # next state is not terminal
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
import numpy as np
import pandas as pd

class VTable:
    
    def __init__(self, actions, reward_decay, alphaW=.1, alphaV=.1, beta=2):

        self.actions = actions
        #self.v = np.random.randint(5, size=(world.n, world.m))
        self.v = pd.DataFrame(columns=self.actions, dtype=np.float64)   # matrix of state-actions values
        self.alphaW = alphaW                    # learning rate for w
        self.alphaV = alphaV                    # learning rate for v
        self.beta = beta
        self.gamma = reward_decay
        self.w = np.zeros(self.q_table.shape[0])              # vector of conditioned reinforcement values, one for each state
        
    def probabilities(self, states):
        p = pd.DataFrame(columns=self.actions, dtype=np.float64)
        for s in range(states):
            initial = np.array((self.v[s, :] * self.beta))
            vexp = np.exp(initial) # vector of exponents
            vexp = vexp/sum(vexp)
            p[s] = vexp

        return p
    
    def action(self, states, state):
        pm = self.probabilities(states)
        pr = pm[state]
        print(self.v)
        choice = np.random.choice(self.v[state, :], p=pr)
        return self.v[state, :].tolist().index(choice)
    
    def learning(self,fromState, action, toState, world):
        totreward = world.values[toState] + self.w[toState]
        deltaV = self.alphaV * (totreward - self.v[fromState, action])
        self.v[fromState, action] += deltaV
        return self

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
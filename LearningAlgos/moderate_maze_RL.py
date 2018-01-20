import numpy as np
import pandas as pd
import math

class QLearningTable:

    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.global_e_greedy = e_greedy
        self.q_table_category = {}
        self.greedy_dict = {}
        self.agent_extra_state = ""
        self.decay_count = 0
        self.max_reward = {}

    def set_prior_qtable(self, key, df_qtable):
        self.q_table_category[key] = df_qtable

    def set_greedy_rule(self, greedy_rate, episode, max_greedy):
        base = 1 - self.global_e_greedy
        target = 1 - max_greedy
        self.epoch_to_update = []
        for ind_greedy in greedy_rate:
            rounds =  math.ceil(math.log(target / base, ind_greedy))
            epos = math.floor(episode/rounds)
            self.epoch_to_update.append(epos)
        self.greedy_rate = greedy_rate
        self.max_greedy = max_greedy

    def update_episode(self, key):
        if key not in self.greedy_dict:
            self.greedy_dict[key] = [1, self.global_e_greedy, self.epoch_to_update[self.decay_count], self.greedy_rate[self.decay_count]]
            if len(self.epoch_to_update) > self.decay_count + 1:
                self.decay_count = self.decay_count + 1
        else:
            obj = self.greedy_dict[key]
            epi = obj[0] + 1
            epsilon = obj[1]
            epoch_update = obj[2]
            if epi != 0 and epi % epoch_update == 0 and epsilon != self.max_greedy:
                greedy_rate = obj[3]
                epsilon = 1 - (1 - epsilon) * greedy_rate
                epsilon = self.max_greedy if epsilon > self.max_greedy else epsilon
                print(key)
                print(epi)
                print(epsilon)
                print()
                # print(epsilon)
            self.greedy_dict[key][0] = epi
            self.greedy_dict[key][1] = epsilon

    def choose_action(self, state, extra_state):
        # print()
        observation = str(state)
        self.check_state_exist(extra_state, observation)
        epsilon = self.global_e_greedy
        if extra_state in self.greedy_dict:
            epsilon = self.greedy_dict[extra_state][1]
        # action selection
        if np.random.uniform() < epsilon:
            # choose best action
            state_action = self.q_table_category[extra_state].ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return int(action)

    def learn(self, _s, extra_s, a, r, _s_, extra_state, is_done):
        reward_coefficient = 1
        virtual_done = False
        if is_done:
            self.agent_extra_state = ""
        elif extra_state != self.agent_extra_state:
            if extra_state != '[]_[]':
                virtual_done = True
            self.update_episode(extra_state)
            self.agent_extra_state = extra_state

        s = str(_s)
        s_ = str(_s_)
        self.check_state_exist(extra_state, s_)
        q_predict = self.q_table_category[extra_s].ix[s, a]
        if virtual_done:
            next_expectation = 0 if extra_state not in self.max_reward else self.max_reward[extra_state]
            q_target = r + next_expectation
            reward_coefficient = self.check_max_reward(extra_s, q_target)
            # print(self.max_reward)
            # print(q_target)
        elif not is_done:
            q_target = r + self.gamma * self.q_table_category[extra_state].ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            reward_coefficient = self.check_max_reward(extra_s, q_target)
            # print(q_target)
        self.q_table_category[extra_s].ix[s, a] += reward_coefficient * self.lr * (q_target - q_predict)  # update

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
            if max_val > r:
                self.max_reward[state_key] = 0.99*self.max_reward[state_key]
                return 1 - math.tanh(-math.log2(r/max_val))*3
            else:
                self.max_reward[state_key] = r
                return 1

    def check_state_exist(self, extra_state, state):
        if extra_state not in self.q_table_category:
            q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
            q_table = q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=q_table.columns,
                    name=state,
                )
            )
            self.q_table_category[extra_state] = q_table
        else:
            q_table = self.q_table_category[extra_state]
            if state not in q_table.index:
                # append new state to q table
                q_table = q_table.append(
                    pd.Series(
                        [0]*len(self.actions),
                        index=q_table.columns,
                        name=state,
                    )
                )
                self.q_table_category[extra_state] = q_table

    def print_list(self, ls):
        str_val = ""
        if len(ls) == 0:
            return "[]"
        for obj in ls:
            str_val += obj
        return str_val

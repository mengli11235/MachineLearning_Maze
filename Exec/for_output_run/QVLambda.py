from MazeEnv.maze_layouts import MazeSmall, MazeLarge, MazeMedium
from LearningAlgos.QV_lambda_RL import QTable, VTable
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv
import math
import statistics
import numpy as np


class QVLambda:
    def __init__(self):
        pass

    def learning(self, epi, max_steps, time_in_ms, _is_render, QL, VL, env):
        rewards_memory = []
        step_reward = []
        rewards = []
        time_array = []
        epo = []
        episode = 0

        while True:
            # initiate the agent
            agent = env.reset()
            reward_in_each_epi = 0
            init_time = time.time()

            for i in range(max_steps):
                # fresh env
                env.render(time_in_ms)

                # RL choose action based on observation
                current_state = str(agent)
                action = QL.choose_action(current_state)

                # RL take action and get next observation and reward
                new_state, extra_state, reward, is_done = env.taking_action(action)
                new_state = np.append(new_state, extra_state)
                reward_in_each_epi += reward
                rewards_memory.append(reward)
                # print(new_state)

                # RL learn from this transition
                QL.learn(VL, current_state, action, reward, str(new_state), is_done)
                VL.update(current_state, reward, str(new_state), is_done)

                # swap observation
                agent = new_state

                # calculate average reward per n steps
                if len(rewards_memory) == 500:
                    step_reward.append(statistics.mean(rewards_memory))
                    rewards_memory = []

                # break inner for loop when end of this episode
                if is_done or i == max_steps-1:
                    # print(episode/epi)
                    print('reward in this epoch: ' + str(reward_in_each_epi))
                    rewards.append(reward_in_each_epi)
                    # print(rewards)
                    time_array.append(format(time.time() - init_time, '.2f'))
                    # print(time_array)
                    epo.append(episode + 1)
                    # print(epo)
                    break
                # print(epi)
                if epi <= 0:
                    break

            epi = epi - 1
            print('remaining epochs: ' + str(epi))
            # print(QL.q_table)
            # if epi <= 2:
            #     print(VL.v)
            #     print(VL.traces)
            # print(QL.epsilon)
            QL.update_episode(epi)
            episode = episode + 1
            if epi <= 0:
                break

        # end of game
        # print('game over')
        if _is_render:
            time.sleep(1)
            env.destroy()

        # QL.q_table.to_csv("temp_q_table.csv", sep=',', encoding='utf-8')
        # print(QL.q_table)
        # plt.plot(epo, rewards)
        # plt.show()
        return step_reward

    def run(self, lambda_, maze_index, epi, episodes, max_steps, reward_gamma, greedy_rule, max_reward_coefficient):
        # set if render the GUI
        is_render = False
        is_demo = False

        # set number of runs
        episodes = episodes
        # animation interval
        interval = 0.005

        # initial position of the agent
        # all position count from 0
        init_pos = [0, 0]

        # initiate maze simulator for learning and running
        if is_demo:
            is_render = True

        if maze_index == 0:
            maze = MazeSmall(init_pos).init_maze(is_render)
        elif maze_index == 1:
            maze = MazeMedium(init_pos).init_maze(is_render)
        elif maze_index == 2:
            maze = MazeLarge(init_pos).init_maze(is_render)

        # initiate QLearner
        actions = list(range(maze.n_actions))
        learning_rate_v = 0.1
        learning_rate_q = 0.1
        reward_gamma = reward_gamma
        greedy = greedy_rule[0]
        lambda_v = lambda_
        QLearner = QTable(actions, learning_rate_q, reward_gamma, greedy, epi)
        Vlearner = VTable(learning_rate_v, reward_gamma, lambda_v)

        # run the simulation of training
        if is_render:
            maze.after(1, self.learning(epi, max_steps, interval, is_render, QLearner, Vlearner, maze))
            maze.mainloop()
        else:
            return self.learning(epi, max_steps, interval, is_render, QLearner, Vlearner, maze)

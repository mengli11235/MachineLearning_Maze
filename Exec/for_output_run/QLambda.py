from MazeEnv.maze_layouts import MazeSmall, MazeLarge, MazeMedium
from LearningAlgos.Q_lambda_RL import QLearningTable
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv
import math
import statistics


class QLambda:
    def __init__(self):
        pass

    def learning(self, epi, time_in_ms, _is_render, QL, env, max_steps):
        rewards_memory = []
        step_reward = []
        rewards = []
        time_array = []
        epo = []
        step_array = []
        training_time = time.time()
        per_5 = math.floor(epi / 20)
        episode = 0

        while True:
            # initiate the agent
            agent, cond = env.reset()
            reward_in_each_epi = 0
            init_time = time.time()
            step = 0

            # if episode % per_5 == 0:
                # print("{} %".format((episode / per_5) * 5))
                # print()

            # initial all zero eligibility trace
            QL.reset_trace()

            for i in range(max_steps):
                # fresh env
                env.render(time_in_ms)

                # RL choose action based on observation
                action = QL.choose_action(agent, cond)

                # RL take action and get next observation and reward
                new_state, new_cond, reward, is_done = env.taking_action(action)
                reward_in_each_epi += reward
                rewards_memory.append(reward)

                # RL learn from this transition
                QL.learn(agent, cond, action, reward, new_state, new_cond, is_done)

                # swap observation
                agent = new_state
                cond = new_cond

                # count step
                step = step + 1
                epi = epi - 1

                # calculate average reward per n steps
                if len(rewards_memory) == 500:
                    step_reward.append(statistics.mean(rewards_memory))
                    rewards_memory = []
                    # print(step_reward)

                # break while loop when end of this episode
                if is_done:
                    rewards.append(reward_in_each_epi)
                    time_array.append(format(time.time() - init_time, '.2f'))
                    step_array.append(step)
                    # step_array = step_counter(step_array)
                    # print(time_array)
                    epo.append(episode + 1)
                    if _is_render:
                        # print(episode/epi)
                        print(reward_in_each_epi)
                        print()
                        # print(epo)
                    break
                # print(epi)
                if epi <= 0:
                    break

            episode = episode + 1

            if epi <= 0:
                print("finish")
                # print(episode)
                break

        # end of game
        # print('game over')
        # # print training time
        # training_time = time.time() - training_time
        # m, s = divmod(training_time, 60)
        # h, m = divmod(m, 60)
        # print("Total training time: %d hr %02d min %02d sec" % (h, m, s))
        #
        # qtable_keys = QL.q_table_category.keys()
        # with open('tmp_data/q_lambda_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(qtable_keys)
        # for key in qtable_keys:
        #     QL.q_table_category[key].to_csv("tmp_data/temp_q_lambda_" + key + ".csv", sep=',', encoding='utf-8')


        # plt.plot(range(len(step_reward)), step_reward)
        # plt.figure(1)
        # plt.plot(epo, rewards)
        # plt.figure(2)
        # plt.plot(epo, step_array)
        # plt.figure(3)
        # plt.plot(epo, [r/s for r, s in zip(rewards, step_array)])
        # plt.show()

        return step_reward

        if _is_render:
            time.sleep(1)
            env.destroy()

    def running(self, epi, time_in_ms, _is_render, QL, env):
        try:
            with open('tmp_data/q_lambda_category.csv', 'r') as f:
                reader = csv.reader(f)
                qtable_keys = list(reader)[0]
                for key in qtable_keys:
                    df = pd.DataFrame.from_csv("tmp_data/temp_q_lambda_" + key + ".csv", sep=',', encoding='utf8')
                    QL.set_prior_qtable(key, df)
                print("set prior q")
        except Exception:
            pass

        for episode in range(epi):
            # initiate the agent
            agent, cond = env.reset()
            reward_in_each_epi = 0

            while True:
                # fresh env
                env.render(time_in_ms)

                # RL choose action based on observation
                action = QL.choose_action(agent, cond)

                # RL take action and get next observation and reward
                new_state, new_cond, reward, is_done = env.taking_action(action)
                reward_in_each_epi += reward

                # swap observation
                agent = new_state
                cond = new_cond

                # break while loop when end of this episode
                if is_done:
                    break

                print(epi)
                if epi <= 0:
                    break

            # episode = episode + 1

            if epi <= 0:
                break

            if _is_render:
                print(reward_in_each_epi)

        # end of game)
        print('game over')
        if _is_render:
            time.sleep(30)
            env.destroy()

    def run(self, lambda_, maze_index, epi, episodes, max_steps, reward_gamma, greedy_rule, max_reward_coefficient):
        # set if render the GUI
        is_render = False
        is_demo = False
        # maximal number of states
        max_steps = max_steps
        # set number of uns
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
        learning_rate = 0.1
        reward_gamma = reward_gamma
        greedy = greedy_rule[0]
        from_lambda_val = lambda_
        to_lambda_val = lambda_
        max_reward_coefficient = max_reward_coefficient
        QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy, from_lambda_val, to_lambda_val,
                                  max_reward_coefficient)
        QLearner.set_greedy_rule(greedy_rule[1], episodes * 0.95, greedy_rule[2])

        # run the training
        if not is_demo:
            if is_render:
                maze.after(1, self.learning(epi, interval, is_render, QLearner, maze))
                maze.mainloop()
            else:
                return self.learning(epi, interval, is_render, QLearner, maze, max_steps)
        # run the simulation of result
        else:
            # Q decision with 99% greedy strategy
            demo_greedy = 0.99
            demo_interval = 0.05
            QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
            self.running(30, demo_interval, True, QRunner, maze)

from MazeEnv.maze_layouts import MazeSmall, MazeLarge, MazeMedium
from LearningAlgos.Sarsa_lambda_RL import SarsaLambda
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv
import math
import statistics


class _SarsaLambda:
    def __init__(self):
        pass

    def learning(self, epi, time_in_ms, _is_render, SL, env, max_steps):
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
            # initial observation
            # observation = env.reset()
            reward_in_epoch = 0
            init_time = time.time()
            step = 0

            # if episode % per_5 == 0:
            #     print("{} %".format((episode / per_5) * 5))
            #     print()

            # initial observation
            agent, cond = env.reset()

            # SL choose action based on observation
            action = SL.choose_action(agent, cond)

            # initial all zero eligibility trace
            SL.reset_trace()

            for i in range(max_steps):
                # fresh env
                env.render(time_in_ms)

                # SL take action and get next observation and reward
                new_state, new_cond, reward, is_done = env.taking_action(action)
                reward_in_epoch += reward
                rewards_memory.append(reward)

                # SL choose action based on next observation
                action_ = SL.choose_action(new_state, new_cond)

                # SL learn from this transition (s, a, r, s, a) ==> Sarsa
                SL.learn(agent, action, reward, new_state, action_, cond, new_cond, is_done)

                # swap observation and action
                agent = new_state
                cond = new_cond
                action = action_

                # count step
                step = step + 1
                epi = epi - 1

                # calculate average reward per n steps
                if len(rewards_memory) == 5000 or is_done:
                    step_reward.append(statistics.mean(rewards_memory))
                    rewards_memory = []

                # break while loop when end of this episode
                if is_done:
                    rewards.append(reward_in_epoch)
                    time_array.append(format(time.time() - init_time, '.2f'))
                    step_array.append(step)
                    epo.append(episode + 1)
                    break

                # print(epi)
                if epi <= 0:
                    break

            episode = episode + 1

            if epi <= 0:
                break

        # end of game
        # print('game over')
        # print training time
        # training_time = time.time() - training_time
        # m, s = divmod(training_time, 60)
        # h, m = divmod(m, 60)
        # print("Total training time: %d hr %02d min %02d sec" % (h, m, s))

        # sarsa_keys = SL.q_table_category.keys()
        # with open('tmp_data/sarsa_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(sarsa_keys)
        # for key in sarsa_keys:
        #     SL.q_table_category[key].to_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',', encoding='utf-8')
            # print(SL.q_table_category[key])

        plt.figure(1)
        plt.plot(epo, rewards)
        # plt.figure(2)
        # plt.plot(epo, step_array)
        plt.show()

        return step_reward

        if _is_render:
            time.sleep(1)
            env.destroy()

    def running(self, epi, time_in_ms, _is_render, SL, env):
        try:
            with open('tmp_data/sarsa_category.csv', 'r') as f:
                reader = csv.reader(f)
                qtable_keys = list(reader)[0]
                for key in qtable_keys:
                    df = pd.DataFrame.from_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',',
                                               encoding='utf8')
                    SL.set_prior_qtable(key, df)
                print("set prior q")
        except Exception:
            pass

        rewards = []
        time_array = []
        epo = []
        for episode in range(epi):
            reward_in_each_epi = 0
            init_time = time.time()

            # initial observation
            agent, cond = env.reset()

            # SL choose action based on observation
            action = SL.choose_action(agent, cond)

            # initial all zero eligibility trace
            SL.reset_trace()

            while True:
                # fresh env
                env.render(time_in_ms)

                # SL take action and get next observation and reward
                new_state, new_cond, reward, is_done = env.taking_action(action)
                reward_in_each_epi += reward

                # SL choose action based on next observation
                action_ = SL.choose_action(new_state, new_cond)

                # swap observation and action
                # agent = new_state
                # cond = new_cond
                action = action_

                # break while loop when end of this episode
                if is_done:
                    rewards.append(reward_in_each_epi)
                    time_array.append(format(time.time() - init_time, '.2f'))
                    epo.append(episode + 1)
                    break

            if _is_render:
                print(reward_in_each_epi)

        # end of game
        print('game over, total rewards gained for each epoch:')
        print(rewards)
        print('time (in sec) spent over epochs:')
        print(time_array)

        if _is_render:
            env.destroy()

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

        # maximal number of states
        max_steps = max_steps

        # initiate maze simulator for learning and running
        if is_demo:
            is_render = True

        if maze_index == 0:
            maze = MazeSmall(init_pos).init_maze(is_render)
        elif maze_index == 1:
            maze = MazeMedium(init_pos).init_maze(is_render)
        elif maze_index == 2:
            maze = MazeLarge(init_pos).init_maze(is_render)

        # initiate SarsaLearner
        actions = list(range(maze.n_actions))
        learning_rate = 0.1
        reward_gamma = reward_gamma

        greedy = greedy_rule[0]
        # lambda_val = 0
        lambda_val = lambda_
        max_reward_coefficient = max_reward_coefficient
        SLearner = SarsaLambda(actions, learning_rate, reward_gamma, greedy, lambda_val, max_reward_coefficient)
        SLearner.set_greedy_rule(greedy_rule[1], episodes * 0.95, greedy_rule[2])

        # run the training
        if not is_demo:
            if is_render:
                maze.after(1, self.learning(epi, interval, is_render, SLearner, maze, max_steps))
                maze.mainloop()
            else:
                return self.learning(epi, interval, is_render, SLearner, maze, max_steps)
        # run the simulation of result
        else:
            # Q decision with 99% greedy strategy
            demo_greedy = 0.99
            demo_interval = 0.05
            SRunner = SarsaLambda(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
            self.running(30, demo_interval, True, SRunner, maze)

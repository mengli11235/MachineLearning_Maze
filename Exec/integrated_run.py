from Exec.for_output_run.QLambda import QLambda
from Exec.for_output_run.SarsaLambda import _SarsaLambda
from Exec.for_output_run.QVLambda import QVLambda
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv


<<<<<<< HEAD
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
                if len(rewards_memory) == 5000 or is_done:
                    step_reward.append(statistics.mean(rewards_memory))
                    rewards_memory = []

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
        return step_reward

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
            time.sleep(500)
            env.destroy()

    def run(self, lambda_, maze_index, epi, unit):
        # set if render the GUI
        is_render = False
        is_demo = False
        # maximal number of states
        max_steps = 1500
        # set number of uns
        episodes = math.floor(epi/unit)
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
        reward_gamma = 0.95
        greedy = 0.4
        from_lambda_val = lambda_
        to_lambda_val = lambda_
        max_reward_coefficient = 0.75
        QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy, from_lambda_val, to_lambda_val,
                                  max_reward_coefficient)
        QLearner.set_greedy_rule([0.9], episodes * 0.95, 0.9)

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
                    epo.append(1)
                    break

                # print(epi)
                if epi <= 0:
                    break

            # episode = episode + 1

            if epi <= 0:
                break
        return step_reward
        # end of game
        print('game over')
        # print training time
        training_time = time.time() - training_time
        m, s = divmod(training_time, 60)
        h, m = divmod(m, 60)
        print("Total training time: %d hr %02d min %02d sec" % (h, m, s))

        sarsa_keys = SL.q_table_category.keys()
        with open('tmp_data/sarsa_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(sarsa_keys)
        for key in sarsa_keys:
            SL.q_table_category[key].to_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',', encoding='utf-8')
            # print(SL.q_table_category[key])

        plt.figure(1)
        plt.plot(epo, rewards)
        plt.figure(2)
        plt.plot(epo, step_array)
        plt.show()

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

    def run(self, lambda_, maze_index, epi, unit):
        # set if render the GUI
        is_render = False
        is_demo = False
        # set number of runs
        episodes = math.floor(epi / unit)
        # animation interval
        interval = 0.005

        # initial position of the agent
        # all position count from 0
        init_pos = [0, 0]

        # maximal number of states
        max_steps = 1500

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
        reward_gamma = 0.95

        greedy = 0.4
        # lambda_val = 0
        lambda_val = lambda_
        max_reward_coefficient = 0.75
        SLearner = SarsaLambda(actions, learning_rate, reward_gamma, greedy, lambda_val, max_reward_coefficient)
        SLearner.set_greedy_rule([0.9], episodes * 0.95, 0.9)

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


class QVLambda:
    def __init__(self):
        pass

    def learning(self, epi, max_steps, time_in_ms, _is_render, QL, VL, env):
        rewards_memory = []
        step_reward = []
        rewards = []
        time_array = []
        epo = []

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
                new_state, reward, is_done = env.taking_action(action)
                reward_in_each_epi += reward
                rewards_memory.append(reward)

                # RL learn from this transition
                QL.learn(VL, current_state, action, reward, str(new_state), is_done)
                VL.update(current_state, reward, str(new_state), is_done)

                # swap observation
                agent = new_state

                epi = epi - 1

                # calculate average reward per n steps
                if len(rewards_memory) == 5000 or is_done:
                    step_reward.append(statistics.mean(rewards_memory))
                    rewards_memory = []

                # break while loop when end of this episode
                if is_done:
                    # print(episode/epi)
                    rewards.append(reward_in_each_epi)
                    # print(rewards)
                    time_array.append(format(time.time() - init_time, '.2f'))
                    # print(time_array)
                    epo.append(1)
                    # print(epo)
                    break
                # print(epi)
                if epi <= 0:
                    break

            if epi <= 0:
                break
        return step_reward
        # end of game
        print('game over')
        if _is_render:
            time.sleep(1)
            env.destroy()

        QL.q_table.to_csv("temp_q_table.csv", sep=',', encoding='utf-8')
        # print(QL.q_table)
        plt.plot(epo, rewards)
        plt.show()

    def run(self, lambda_, maze_index, epi, unit):
        # set if render the GUI
        is_render = False
        is_demo = False
        # set number of runs
        episodes = math.floor(epi / unit)
        # animation interval
        interval = 0.005

        # initial position of the agent
        # all position count from 0
        init_pos = [0, 0]

        # maximal number of states
        max_steps = 1500

        # initiate maze simulator for learning and running
        if is_demo:
            is_render = True

        if maze_index == 0:
            maze = MazeSmallQV(init_pos).init_maze(is_render)
        elif maze_index == 1:
            maze = MazeMediumQV(init_pos).init_maze(is_render)
        elif maze_index == 2:
            maze = MazeLargeQV(init_pos).init_maze(is_render)

        # initiate QLearner
        actions = list(range(maze.n_actions))
        learning_rate_v = 0.1
        learning_rate_q = 0.1
        reward_gamma = 0.95
        greedy = 0.6
        lambda_v = lambda_
        QLearner = QTable(actions, learning_rate_q, reward_gamma, greedy)
        Vlearner = VTable(learning_rate_v, reward_gamma, lambda_v)

        # run the simulation of training
        if is_render:
            maze.after(1, self.learning(epi, max_steps, interval, is_render, QLearner, Vlearner, maze))
            maze.mainloop()
        else:
            return self.learning(epi, max_steps, interval, is_render, QLearner, Vlearner, maze)


if __name__ == "__main__":
    # lambda_arr = [0]
    lambda_arr = [0, 0.5, 0.8]
    # f1 = QLambda().run
=======
def single_run(func, lmb, maze_index, total_epi, episodes, algo_type, simulation, max_steps, reward_gamma, greedy_rule, max_reward_coefficient):
    print("start", algo_type)
    min_steps = 999999
    f_avg_rw_arr = []
    f_avg_rw = []
    for ix in range(simulation):
        p = func(lmb, maze_index, total_epi, episodes, max_steps, reward_gamma, greedy_rule, max_reward_coefficient)
        f_avg_rw_arr.append(p)
        if min_steps > len(p):
            min_steps = len(p)
    for idx in range(min_steps):
        sum_val = 0
        length = len(f_avg_rw_arr)
        for ix in range(length):
            sum_val += f_avg_rw_arr[ix][idx]
        f_avg_rw.append(sum_val / length)

    with open('tmp_data/'+ algo_type + str(index) + '.csv',
              'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(f_avg_rw)


if __name__ == "__main__":
    algo_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    lambda_arr = [0]
    # lambda_arr = [0, 0.5, 0.8]
    f1 = QLambda().run
>>>>>>> df928283e98e051066198c1137880376aaf4d0eb
    f2 = _SarsaLambda().run
    # f3 = QVLambda().run

<<<<<<< HEAD
    simulation = 30
    maze_index = 2
    total_epi = 25000
    steps_epoch = 200
=======
    simulation = 1
    maze_index = 0

    # base parameter values
    total_epi = 15000
    episodes = 250
    max_steps = 500
    reward_gamma = 0.95
    greedy_rule = [0.4, [0.9], 0.9]
    max_reward_coefficient = 0.8

    para_names = ["total_epi", "episodes", "max_steps", "reward_gamma", "greedy_rule", "max_reward_coefficient"]
    parameter_dict = {
        algo_names[0]: [{
            para_names[0]: total_epi,
            para_names[1]: 300,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: greedy_rule,
            para_names[5]: 0.8
        }],
        algo_names[1]: [{
            para_names[0]: total_epi,
            para_names[1]: 300,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: greedy_rule,
            para_names[5]: 0.8
        }],
        algo_names[2]: [{
            para_names[0]: total_epi,
            para_names[1]: 180,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: [0.6, [0.9], 0.9],
            para_names[5]: 0.8
        }]
    }
>>>>>>> df928283e98e051066198c1137880376aaf4d0eb

    for index in range(len(lambda_arr)):
        lmb = lambda_arr[index]

<<<<<<< HEAD
=======
        _parameters = parameter_dict[algo_names[0]][index]
        single_run(f1, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[0],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        _parameters = parameter_dict[algo_names[1]][index]
        single_run(f2, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[1],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        _parameters = parameter_dict[algo_names[2]][index]
        single_run(f3, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[2],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

>>>>>>> df928283e98e051066198c1137880376aaf4d0eb
        # print("start Q lambda")
        # min_steps = 999999
        # f1_avg_rw_arr = []
        # f1_avg_rw = []
        # for ix in range(simulation):
        #     p = f1(lmb, maze_index, total_epi, steps_epoch)
        #     f1_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f1_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f1_avg_rw_arr[ix][idx]
        #     f1_avg_rw.append(sum_val/length)
        #
        # with open('tmp_data/q_lambda_avg_reward' + str(index) + '.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f1_avg_rw)
        #
<<<<<<< HEAD
        print("start SARSA lambda")
        min_steps = 999999
        f2_avg_rw_arr = []
        f2_avg_rw = []
        for ix in range(simulation):
            p = f2(lmb, maze_index, total_epi, steps_epoch)
            f2_avg_rw_arr.append(p)
            if min_steps > len(p):
                min_steps = len(p)
        for idx in range(min_steps):
            sum_val = 0
            length = len(f2_avg_rw_arr)
            for ix in range(length):
                sum_val += f2_avg_rw_arr[ix][idx]
            f2_avg_rw.append(sum_val / length)

        with open('tmp_data/sarsa_lambda_avg_reward' + str(index) + '.csv',
                  'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(f2_avg_rw)
=======
        # print("start SARSA lambda")
        # min_steps = 999999
        # f2_avg_rw_arr = []
        # f2_avg_rw = []
        # for ix in range(simulation):
        #     p = f2(lmb, maze_index, total_epi, steps_epoch)
        #     f2_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f2_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f2_avg_rw_arr[ix][idx]
        #     f2_avg_rw.append(sum_val / length)
        #
        # with open('tmp_data/sarsa_lambda_avg_reward' + str(index) + '.csv',
        #           'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f2_avg_rw)
>>>>>>> df928283e98e051066198c1137880376aaf4d0eb

        # print("start QV lambda")
        # min_steps = 999999
        # f3_avg_rw_arr = []
        # f3_avg_rw = []
        # for ix in range(simulation):
        #     p = f3(lmb, maze_index, total_epi, steps_epoch)
        #     f3_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f3_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f3_avg_rw_arr[ix][idx]
        #     f3_avg_rw.append(sum_val / length)
        #
        # with open('tmp_data/qv_lambda_avg_reward' + str(index) + '.csv',
        #           'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f3_avg_rw)

        print("finish lambda", str(lmb))
        # plt.plot(range(len(f1_avg_rw)), f1_avg_rw)
        # plt.plot(range(len(f2_avg_rw)), f2_avg_rw)
        # plt.plot(range(len(f3_avg_rw)), f3_avg_rw)
        # plt.show()
        #
        # for ix in range(simulation):
        #     f2(lmb)
        #
        # for ix in range(simulation):
        #     f3(lmb)


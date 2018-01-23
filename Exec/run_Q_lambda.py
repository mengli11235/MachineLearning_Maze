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


def step_counter(_current_array=-1, _length=30):
    if _current_array == -1:
        return _current_array
    else:
        _sum = 0
        if len(_current_array) >= _length:
            start_index = len(_current_array) - _length
            for x in range(0, _length):
                _sum += _current_array[start_index + x]
            new_mean = _sum/_length
            _current_array[start_index] = new_mean
            return _current_array
        return _current_array


def learning(epi, time_in_ms, _is_render, QL, env, max_steps):
    rewards = []
    time_array = []
    epo = []
    step_array = []
    training_time = time.time()
    per_5 = math.floor(epi/20)

    for episode in range(epi):
        # initiate the agent
        agent, cond = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()
        step = 0

        if episode%per_5 == 0:
            print("{} %".format((episode/per_5)*5))
            print()

        # initial all zero eligibility trace
        QL.reset_trace()

        for step in range(max_steps):
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(agent, cond)

            # RL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # RL learn from this transition
            QL.learn(agent, cond, action, reward, new_state, new_cond, is_done)

            # swap observation
            agent = new_state
            cond = new_cond

            # count step
            step = step + 1

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_each_epi)
                time_array.append(format(time.time() - init_time, '.2f'))
                step_array.append(step)
                # step_array = step_counter(step_array)
                # print(time_array)
                epo.append(episode+1)
                if _is_render:
                    # print(episode/epi)
                    print(reward_in_each_epi)
                    print()
                    # print(epo)
                break

    # end of game
    print('game over')
    # print training time
    training_time = time.time() - training_time
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    print("Total training time: %d hr %02d min %02d sec" % (h, m, s))

    qtable_keys = QL.q_table_category.keys()
    with open('tmp_data/q_lambda_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(qtable_keys)
    for key in qtable_keys:
        QL.q_table_category[key].to_csv("tmp_data/temp_q_lambda_" + key + ".csv", sep=',', encoding='utf-8')
        # print(QL.q_table_category[key])

    plt.figure(1)
    plt.plot(epo, rewards)
    plt.figure(2)
    plt.plot(epo, step_array)
    # plt.figure(3)
    # plt.plot(epo, [r/s for r, s in zip(rewards, step_array)])
    plt.show()

    if _is_render:
        time.sleep(1)
        env.destroy()


def running(epi, time_in_ms, _is_render, QL, env):
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

        if _is_render:
            print(reward_in_each_epi)

    # end of game)
    print('game over')
    if _is_render:
        time.sleep(500)
        env.destroy()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    is_demo = False
    # set number of uns
    episodes = 300
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

    # maze = MazeSmall(init_pos).init_maze(is_render)
    maze = MazeMedium(init_pos).init_maze(is_render)
    # maze = MazeLarge(init_pos).init_maze(is_render)

    # initiate QLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.4
    from_lambda_val = 0.5
    to_lambda_val = 0.5
    max_reward_coefficient = 0.75
    QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy, from_lambda_val, to_lambda_val, max_reward_coefficient)
    QLearner.set_greedy_rule([0.9], episodes*0.95, 0.9)

    # run the training
    if not is_demo:
        if is_render:
            maze.after(1, learning(episodes, interval, is_render, QLearner, maze))
            maze.mainloop()
        else:
            learning(episodes, interval, is_render, QLearner, maze, max_steps)
    # run the simulation of result
    else:
        # Q decision with 99% greedy strategy
        demo_greedy = 0.99
        demo_interval = 0.05
        QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
        running(30, demo_interval, True, QRunner, maze)

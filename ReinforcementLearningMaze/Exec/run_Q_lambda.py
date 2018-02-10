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


def learning(total_steps, time_in_ms, _is_render, QL, env, max_steps):
    # store the data of learning process for the plotting later
    episode = 0
    step_counter = 0
    rewards = []
    time_array = []
    epo = []
    step_array = []
    training_time = time.time()
    per_5 = math.floor(total_steps/20)

    while True:
        # initialize the agent
        agent, cond = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()
        force_exit = False

        # reset eligibility trace to zero
        QL.reset_trace()

        for step in range(max_steps):
            step_counter = step_counter + 1
            if step_counter % per_5 == 0:
                print("{} %".format((step_counter / per_5) * 5))
                print()

            # refresh rendering
            env.render(time_in_ms)

            # choose action based on current state
            action = QL.choose_action(agent, cond)

            # take action and get next state and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            if step == max_steps - 1:
                force_exit = True
            # learning
            QL.learn(agent, cond, action, reward, new_state, new_cond, is_done, force_exit)

            # assign to update variables
            agent = new_state
            cond = new_cond

            # break the loop
            if is_done or step_counter >= total_steps:
                if _is_render:
                    print(reward_in_each_epi)
                    print()
                break

        episode = episode + 1
        rewards.append(reward_in_each_epi)
        time_array.append(format(time.time() - init_time, '.2f'))
        step_array.append(step)
        epo.append(episode + 1)

        if step_counter >= total_steps:
            break

    # end of game
    print('game over')
    # print training time
    training_time = time.time() - training_time
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    print("Total training time: %d hr %02d min %02d sec" % (h, m, s))

    # store the table as csv
    qtable_keys = QL.q_table_category.keys()
    with open('tmp_data/q_lambda_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(qtable_keys)
    for key in qtable_keys:
        QL.q_table_category[key].to_csv("tmp_data/temp_q_lambda_" + key + ".csv", sep=',', encoding='utf-8')

    # plot the learning progress
    axes = plt.gca()
    axes.set_ylim([-1000, 1000])
    plt.figure(1)
    plt.plot(epo[:-1], rewards[:-1])
    plt.ylabel("rewards")
    plt.xlabel("epoches")
    plt.title("small maze rewards: Q_lambda(lambda=0.8)")

    plt.figure(2)
    plt.plot(epo[:-1], step_array[:-1])
    plt.ylabel("steps")
    plt.xlabel("epoches")
    plt.title("small maze steps: Q_lambda(lambda=0.8)")
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
    # set if render the GUI of learning
    is_render = False

    # when is_demo set as True, no learning but running GUI to show the learning outcome
    is_demo = False

    # set number of total steps
    total_steps = 5000  # 60000 for medium, 5000(0.5,0.8), 6000(0) for simple

    # animation interval
    interval = 0.005
    # maximal number of states
    max_steps = 150  # 1000 for medium, 150(0.5,0.8), 200(0) for simple

    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True

    maze = MazeSmall(init_pos).init_maze(is_render)
    # maze = MazeMedium(init_pos).init_maze(is_render)

    # initialize QLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.9
    lambda_val = 0.8
    max_reward_coefficient = 0.75
    QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy, lambda_val, max_reward_coefficient)
    QLearner.set_greedy_rule([0.9], 50, greedy)

    # run the training
    if not is_demo:
        if is_render:
            maze.after(1, learning(total_steps, interval, is_render, QLearner, maze, max_steps))
            maze.mainloop()
        else:
            learning(total_steps, interval, is_render, QLearner, maze, max_steps)
    # run the simulation of result
    else:
        # Q decision with 99% greedy strategy
        demo_greedy = 0.99
        demo_interval = 0.05
        QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
        running(30, demo_interval, True, QRunner, maze)
